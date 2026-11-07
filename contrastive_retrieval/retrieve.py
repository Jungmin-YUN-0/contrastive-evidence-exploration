from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .gold import build_doc_index_maps, extract_gold_indices

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "data"

QUERY_EXPANSION_PROMPT = """
Generate search queries to retrieve missing evidence needed to answer the question.

Rules:
- Generate 2 to 4 short search queries.
- Each query should include a concrete entity from the question or retrieved evidence.
- Prefer queries that connect retrieved evidence to an unresolved part of the original question.
- Avoid generic or overly broad topic queries.
- Return JSON only, without explanations.

Question:
{question}

Previously retrieved evidence:
{evidence_block}

Return format:
{{
  "queries": [
    "...",
    "...",
    "..."
  ]
}}
"""


CANDIDATE_SUMMARY_PROMPT = """
Generate a summary of the retrieved documents to answer the question and current probe queries.

Rules:

* Use only the retrieved source documents.
* Include only information relevant to the question and current probe queries.
* Mention explicit entities and relations whenever possible.
* Keep the summary within {summary_max_words} words.
* Return the summary only, without explanations.

Question:
{question}

Probe queries:
{probe_queries}

Retrieved source documents:
{source_documents}

Summary:
"""


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def normalize_doc(title: str, text: Any) -> str:
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return f"{title}\n{text or ''}"


def parse_generated_queries(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("queries", [])
    elif isinstance(raw, str):
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if match is None:
                return []
            try:
                obj = json.loads(match.group(0))
            except Exception:
                return []
        items = (
            obj.get("queries", [])
            if isinstance(obj, dict)
            else obj if isinstance(obj, list) else []
        )
    else:
        return []

    if not isinstance(items, list):
        return []
    queries = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        query = re.sub(r"\s+", " ", item).strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def format_evidence_block(evidence_records: List[Dict[str, Any]]) -> str:
    if not evidence_records:
        return " "
    return "\n".join(
        f"{idx}. {record['title']}: {record['snippet']}"
        for idx, record in enumerate(evidence_records, start=1)
    )


def truncate_summary_words(text: str, max_words: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_words <= 0:
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]).rstrip()


def format_probe_queries_for_summary(probe_queries: List[str]) -> str:
    if not probe_queries:
        return "None"

    return "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(probe_queries, start=1)
    )


def extract_previous_summary(evidence_records: List[Dict[str, Any]]) -> str:
    if not evidence_records:
        return "None"

    for record in evidence_records:
        if record.get("doc_idx") == -1:
            summary = re.sub(r"\s+", " ", record.get("snippet", "")).strip()
            return summary if summary else "None"

    return "None"


def format_source_documents_for_summary(
    doc_idxs: List[int],
    docs: List[str],
    snippet_chars: int,
) -> str:

    if not doc_idxs:
        return "None"

    records = []

    for rank, doc_idx in enumerate(doc_idxs, start=1):
        title, _, text = docs[int(doc_idx)].partition("\n")
        text = re.sub(r"\s+", " ", text).strip()

        if snippet_chars > 0 and len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + "..."

        records.append(f"[{rank}] Title: {title.strip()}\nText: {text}")

    return "\n\n".join(records)


def summary_to_evidence_record(
    summary: str,
    round_idx: int,
    new_doc_count: int,
    max_words: int,
) -> List[Dict[str, Any]]:

    summary = truncate_summary_words(summary, max_words=max_words)

    if not summary:
        return []

    return [
        {
            "doc_idx": -1,
            "title": f"Updated evidence summary after round {round_idx}",
            "snippet": summary,
            "new_doc_count": int(new_doc_count),
        }
    ]


def select_docs_for_summary_update(
    doc_idxs: List[int],
    doc_scores: Dict[int, float],
    top_k: int,
) -> List[int]:

    if not doc_idxs:
        return []

    if top_k <= 0:
        return list(doc_idxs)

    return sorted(
        doc_idxs,
        key=lambda idx: doc_scores.get(int(idx), float("-inf")),
        reverse=True,
    )[:top_k]


def average_precision_at_k(
    retrieved: List[int], relevant_set: Set[int], k: int
) -> float:
    num_rel = 0
    score = 0.0
    for idx in range(min(k, len(retrieved))):
        if retrieved[idx] in relevant_set:
            num_rel += 1
            score += num_rel / (idx + 1)
    denom = min(len(relevant_set), k)
    return score / denom if denom > 0 else 0.0


def evaluate_results(
    docs: List[str],
    queries: List[str],
    samples: List[Dict[str, Any]],
    rankings: List[List[int]],
    payloads: List[Dict[str, Any]],
    ks: List[int],
    topk: int,
) -> Dict[str, Any]:
    if not (len(queries) == len(samples) == len(rankings) == len(payloads)):
        raise ValueError("Question, sample, ranking, and payload counts must match.")
    if not samples:
        raise ValueError("Cannot evaluate an empty query set.")
    text_to_idx, title_to_idx = build_doc_index_maps(docs)
    gold_sets = [extract_gold_indices(sample, text_to_idx, title_to_idx) for sample in samples]
    results: Dict[str, Any] = {"metrics": {}, "per_query": []}

    def calc_for_k(k: int) -> Dict[str, float]:
        hits = 0
        recalls = []
        aps = []
        mrrs = []
        marrs = []
        ndcgs = []
        supporting_all_count = 0

        for qi, sample in enumerate(samples):
            retrieved = rankings[qi][:k]
            gold_indices = gold_sets[qi]
            num_relevant = len(gold_indices)
            num_found = sum(doc_idx in gold_indices for doc_idx in retrieved)

            hits += int(num_found > 0)
            recalls.append(num_found / num_relevant if num_relevant > 0 else 0.0)
            aps.append(average_precision_at_k(retrieved, gold_indices, k))

            rr = 0.0
            for rank, doc_idx in enumerate(retrieved, start=1):
                if doc_idx in gold_indices:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            sum_rr = sum(
                1.0 / rank
                for rank, doc_idx in enumerate(retrieved, start=1)
                if doc_idx in gold_indices
            )
            marrs.append(sum_rr / num_relevant if num_relevant > 0 else 0.0)

            dcg = sum(
                (1.0 if doc_idx in gold_indices else 0.0) / np.log2(rank + 1)
                for rank, doc_idx in enumerate(retrieved, start=1)
            )
            ideal_rel = min(num_relevant, k)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_rel + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            if num_relevant > 0 and set(retrieved).issuperset(gold_indices):
                supporting_all_count += 1

        num_queries = len(samples)
        return {
            "hits": hits / num_queries,
            "recall": float(np.mean(recalls)),
            "MAP": float(np.mean(aps)),
            "MRR": float(np.mean(mrrs)),
            "MARR": float(np.mean(marrs)),
            "nDCG": float(np.mean(ndcgs)),
            "supporting_all_fraction": supporting_all_count / num_queries,
        }

    for k in sorted(set(ks + [topk])):
        results["metrics"][f"@{k}"] = calc_for_k(k)

    for qi, query in enumerate(queries):
        top_indices = rankings[qi][:topk]
        results["per_query"].append(
            {
                "id": samples[qi].get("id", samples[qi].get("_id")),
                "query": query,
                "top_indices": top_indices,
                "top_texts": [docs[idx] for idx in top_indices],
                **payloads[qi],
            }
        )
    return results


class E5QueryEncoder:
    def __init__(
        self, model_name: str, device: str, batch_size: int, max_seq_length: int
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length > 0:
            self.model.max_seq_length = max_seq_length

    def encode_queries(
        self, texts: List[str], show_progress_bar: bool = False
    ) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        prefixed = [f"query: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)


def _l2_normalize(vecs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=axis, keepdims=True)
    return vecs / np.maximum(norm, eps)


def coverage_aware_greedy_ranking(
    candidate_doc_idxs: List[int],
    relevance_scores: np.ndarray,
    probe_doc_scores: np.ndarray,
    doc_embeddings_p: np.ndarray,
    top_k: int,
    lambda_cov: float,
    lambda_red: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    num_candidates = len(candidate_doc_idxs)
    if num_candidates == 0:
        return [], []
    if relevance_scores.shape != (num_candidates,):
        raise ValueError("Relevance score shape does not match candidate count.")
    if probe_doc_scores.ndim != 2 or probe_doc_scores.shape[1] != num_candidates:
        raise ValueError("Probe score shape does not match candidate count.")

    greedy_k = min(max(0, top_k), num_candidates)
    candidate_doc_idxs_arr = np.asarray(candidate_doc_idxs, dtype=np.int64)
    candidate_vecs = doc_embeddings_p[candidate_doc_idxs_arr]
    selected_offsets: List[int] = []
    selection_records: List[Dict[str, Any]] = []
    available = np.ones(num_candidates, dtype=bool)
    coverage = np.zeros(probe_doc_scores.shape[0], dtype=np.float32)

    for selection_rank in range(1, greedy_k + 1):
        available_offsets = np.flatnonzero(available)
        coverage_gains = np.maximum(
            0.0,
            probe_doc_scores[:, available_offsets] - coverage[:, None],
        ).sum(axis=0)

        if selected_offsets:
            selected_vecs = candidate_vecs[np.asarray(selected_offsets, dtype=np.int64)]
            max_similarities = np.max(
                candidate_vecs[available_offsets] @ selected_vecs.T, axis=1
            )
        else:
            max_similarities = np.zeros(len(available_offsets), dtype=np.float32)

        marginal_gains = (
            relevance_scores[available_offsets]
            + lambda_cov * coverage_gains
            - lambda_red * max_similarities
        )
        local_best = int(np.argmax(marginal_gains))
        best_offset = int(available_offsets[local_best])

        selected_offsets.append(best_offset)
        available[best_offset] = False
        coverage = np.maximum(coverage, probe_doc_scores[:, best_offset])
        selection_records.append(
            {
                "rank": selection_rank,
                "doc_idx": int(candidate_doc_idxs[best_offset]),
                "marginal_gain": float(marginal_gains[local_best]),
                "relevance_score": float(relevance_scores[best_offset]),
                "coverage_gain": float(coverage_gains[local_best]),
                "max_similarity": float(max_similarities[local_best]),
            }
        )

    relevance_order = np.argsort(-relevance_scores, kind="stable")
    selected_set = set(selected_offsets)
    full_order = selected_offsets + [
        int(offset) for offset in relevance_order if int(offset) not in selected_set
    ]
    return full_order, selection_records


def normalize_query_key(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def build_round_probe_queries(question, generated_queries, round_idx, seen_query_keys):
    candidates = ([question] if round_idx == 0 else []) + generated_queries
    probes = []
    for query in candidates:
        key = normalize_query_key(query)
        seen_query_keys.add(key)
        probes.append(re.sub(r"\s+", " ", query).strip())
    return probes


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(max(0, k), len(scores))
    if not k:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[: k - len(above)]
    indices = np.concatenate([above, tied])
    return indices[np.lexsort((indices, -scores[indices]))]


def _masked_softmax(logits, mask, axis=-1):
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), logits.shape)
    masked = np.where(mask, logits, -np.inf)
    maximum = np.max(masked, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weights = np.exp(masked - maximum)
    return weights / np.maximum(weights.sum(axis=axis, keepdims=True), 1e-12)


def rerank_candidates_with_bridge(
    probe_query_embs_q,
    candidate_doc_idxs,
    doc_embeddings_p,
    contrastive_query_embeddings_q,
    contrastive_query_embeddings_p,
    valid_query_mask,
    tau=1.0,
    alpha=0.5,
    beta=0.5,
    facet_representation="dual",
    no_refinement=False,
):

    if facet_representation not in {"dual", "query"}:
        raise ValueError("facet_representation must be dual or query")
    probes = np.asarray(probe_query_embs_q, dtype=np.float32)
    probes = probes.reshape(-1, doc_embeddings_p.shape[1])
    indices = np.asarray(candidate_doc_idxs, dtype=np.int64)
    base_matrix = probes @ doc_embeddings_p[indices].T
    if not len(indices) or not len(probes):
        zeros = np.zeros(len(indices), dtype=np.float32)
        return zeros.copy(), zeros.copy(), zeros.copy(), base_matrix
    refined_matrix = base_matrix.copy()
    if not no_refinement:
        for offset, doc_idx in enumerate(indices):
            valid = np.asarray(valid_query_mask[doc_idx], dtype=bool)
            if not valid.any():
                continue
            facet_q = contrastive_query_embeddings_q[doc_idx][valid]
            facet_update = (
                contrastive_query_embeddings_p[doc_idx][valid]
                if facet_representation == "dual"
                else facet_q
            )
            logits = tau * (probes @ facet_q.T)
            weights = _masked_softmax(logits, np.ones_like(logits, dtype=bool))
            refined_doc = _l2_normalize(
                doc_embeddings_p[doc_idx] + alpha * (weights @ facet_update)
            )
            refined_matrix[:, offset] = np.sum(probes * refined_doc, axis=1)
    base_scores = np.max(base_matrix, axis=0)
    refined_scores = np.max(refined_matrix, axis=0)
    scores = beta * base_scores + (1.0 - beta) * refined_scores
    return scores, base_scores, refined_scores, refined_matrix


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_cache(path):
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in cache: {path}")
    return value


def save_json_cache(path, cache):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def generation_cache_key(kind, prompt, args):
    settings = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "out",
            "query_cache",
            "ks",
            "payload_initial_topk",
            "payload_score_topk",
        }
    }
    return stable_hash(
        {"version": 1, "kind": kind, "prompt": prompt, "settings": settings}
    )


class VLLMTextGenerator:

    def __init__(self, args, kind):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        options = dict(
            model=args.llm_model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.vllm_dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
        )
        if args.max_model_len > 0:
            options["max_model_len"] = args.max_model_len
        self.llm = LLM(**options)
        prefix = "summary" if kind == "summary" else "llm"
        self.sampling_params = SamplingParams(
            temperature=getattr(args, f"{prefix}_temperature"),
            top_p=getattr(args, f"{prefix}_top_p"),
            max_tokens=getattr(args, f"{prefix}_max_tokens"),
            seed=args.seed,
            n=1,
        )

    def generate(self, prompts):
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        generated = self.llm.generate(
            rendered, sampling_params=self.sampling_params, use_tqdm=True
        )
        return [
            output.outputs[0].text.strip() if output.outputs else ""
            for output in generated
        ]


def cached_generate(prompts, args, cache_path, kind):
    cache = load_json_cache(cache_path)
    keys = [generation_cache_key(kind, prompt, args) for prompt in prompts]
    missing = list(
        dict.fromkeys(key for key in keys if not isinstance(cache.get(key), str))
    )
    if missing:
        prompt_by_key = dict(zip(keys, prompts))
        generator = VLLMTextGenerator(args, kind)
        batch_size = (
            args.summary_batch_size if kind == "summary" else args.generation_batch_size
        )
        try:
            for start in range(0, len(missing), batch_size):
                batch_keys = missing[start : start + batch_size]
                outputs = generator.generate([prompt_by_key[key] for key in batch_keys])
                if len(outputs) != len(batch_keys):
                    raise RuntimeError("LLM returned an unexpected number of outputs")
                cache.update(zip(batch_keys, outputs))
                save_json_cache(cache_path, cache)
        finally:
            del generator
            release_cuda_memory()
    return [cache[key] for key in keys]


def load_or_generate_probe_round(
    questions, evidence_records_per_question, round_idx, args, cache_path
):
    if len(questions) != len(evidence_records_per_question):
        raise ValueError("Question and evidence counts must match")
    if args.skip_query_expansion:
        return [[] for _ in questions]
    prompts = [
        QUERY_EXPANSION_PROMPT.format(
            question=question,
            evidence_block=format_evidence_block(evidence),
        )
        for question, evidence in zip(questions, evidence_records_per_question)
    ]
    print(f"[Explore] round={round_idx + 1}, questions={len(prompts)}")
    return [
        parse_generated_queries(raw)
        for raw in cached_generate(prompts, args, cache_path, "probe")
    ]


def load_or_generate_incremental_summaries(
    questions,
    previous_evidence_records_per_question,
    current_probe_queries_per_question,
    current_round_doc_idxs_per_question,
    docs,
    round_idx,
    args,
    cache_path,
):
    if not (
        len(questions)
        == len(previous_evidence_records_per_question)
        == len(current_probe_queries_per_question)
        == len(current_round_doc_idxs_per_question)
    ):
        raise ValueError("Summary input counts must match")
    results = [list(evidence) for evidence in previous_evidence_records_per_question]
    prompts, indices = [], []
    for index, question in enumerate(questions):
        doc_idxs = current_round_doc_idxs_per_question[index]
        if not doc_idxs:
            continue
        indices.append(index)
        prompts.append(
            CANDIDATE_SUMMARY_PROMPT.format(
                question=question,
                probe_queries=format_probe_queries_for_summary(
                    current_probe_queries_per_question[index]
                ),
                source_documents=format_source_documents_for_summary(
                    doc_idxs, docs, args.summary_source_snippet_chars
                ),
                summary_max_words=args.summary_max_words,
            )
        )
    summaries = cached_generate(prompts, args, cache_path, "summary") if prompts else []
    for index, summary in zip(indices, summaries):
        results[index] = summary_to_evidence_record(
            summary,
            round_idx,
            len(current_round_doc_idxs_per_question[index]),
            args.summary_max_words,
        )
    return results


def encode_probe_round(
    probe_queries_per_item: List[List[str]],
    args: argparse.Namespace,
    round_idx: int,
) -> List[np.ndarray]:
    flat_probe_queries = [
        query for probes in probe_queries_per_item for query in probes
    ]
    if not flat_probe_queries:
        return [np.zeros((0, 0), dtype=np.float32) for _ in probe_queries_per_item]

    print(
        f"[Dense][Round {round_idx}] encoding {len(flat_probe_queries)} probe queries..."
    )
    encoder = E5QueryEncoder(
        model_name=args.dense_model,
        device=args.encoder_device,
        batch_size=args.encoder_batch_size,
        max_seq_length=args.encoder_max_seq_length,
    )
    flat_embeddings = encoder.encode_queries(flat_probe_queries, show_progress_bar=True)
    del encoder
    release_cuda_memory()

    embeddings_per_item = []
    offset = 0
    for probes in probe_queries_per_item:
        embeddings_per_item.append(flat_embeddings[offset : offset + len(probes)])
        offset += len(probes)
    return embeddings_per_item


def run_retrieval(
    docs: List[str],
    queries: List[str],
    tensors: Dict[str, np.ndarray],
    args: argparse.Namespace,
    max_k: int,
    query_cache: Optional[Path],
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    candidate_topk = args.candidate_topk
    num_rounds = 1 if args.ablation == "no_exploration" else args.num_rounds
    doc_embeddings_p = tensors["doc_embeddings_p"]
    num_queries = len(queries)

    accumulated_probe_queries: List[List[str]] = [[] for _ in queries]
    accumulated_probe_embeddings: List[List[np.ndarray]] = [[] for _ in queries]
    seen_probe_query_keys: List[Set[str]] = [set() for _ in queries]

    candidate_doc_idxs_per_item: List[List[int]] = [[] for _ in queries]
    seen_docs_per_item: List[Set[int]] = [set() for _ in queries]

    retrieval_rounds_per_item: List[List[Dict[str, Any]]] = [[] for _ in queries]

    evidence_records_per_question: List[List[Dict[str, Any]]] = [[] for _ in queries]

    for round_idx in range(num_rounds):
        generated_queries_per_item = load_or_generate_probe_round(
            questions=queries,
            evidence_records_per_question=evidence_records_per_question,
            round_idx=round_idx,
            args=args,
            cache_path=query_cache,
        )

        round_probe_queries_per_item = [
            build_round_probe_queries(
                question=question,
                generated_queries=generated_queries,
                round_idx=round_idx,
                seen_query_keys=seen_probe_query_keys[qi],
            )
            for qi, (question, generated_queries) in enumerate(
                zip(queries, generated_queries_per_item)
            )
        ]

        round_probe_embeddings_per_item = encode_probe_round(
            probe_queries_per_item=round_probe_queries_per_item,
            args=args,
            round_idx=round_idx,
        )

        current_round_doc_idxs_per_item: List[List[int]] = [[] for _ in queries]
        current_round_seen_docs_per_item: List[Set[int]] = [set() for _ in queries]
        current_round_doc_scores_per_item: List[Dict[int, float]] = [
            {} for _ in queries
        ]

        for qi, (probe_queries, probe_embeddings) in enumerate(
            zip(round_probe_queries_per_item, round_probe_embeddings_per_item)
        ):
            route_records = []

            for probe_query, probe_emb in zip(probe_queries, probe_embeddings):
                scores = doc_embeddings_p @ probe_emb
                ranking = topk_indices(scores, candidate_topk).tolist()

                route_records.append(
                    {
                        "query": probe_query,
                        "top_indices": ranking[: args.payload_initial_topk],
                        "top_scores": [
                            float(scores[idx])
                            for idx in ranking[: args.payload_initial_topk]
                        ],
                    }
                )

                accumulated_probe_queries[qi].append(probe_query)
                accumulated_probe_embeddings[qi].append(probe_emb)

                for doc_idx in ranking:
                    doc_idx = int(doc_idx)
                    score = float(scores[doc_idx])

                    # Accumulated candidate pool for final reranking.
                    if doc_idx not in seen_docs_per_item[qi]:
                        seen_docs_per_item[qi].add(doc_idx)
                        candidate_doc_idxs_per_item[qi].append(doc_idx)

                    # Current-round retrieved passages for incremental summary update.
                    current_round_doc_scores_per_item[qi][doc_idx] = max(
                        current_round_doc_scores_per_item[qi].get(
                            doc_idx, float("-inf")
                        ),
                        score,
                    )

                    if doc_idx not in current_round_seen_docs_per_item[qi]:
                        current_round_seen_docs_per_item[qi].add(doc_idx)
                        current_round_doc_idxs_per_item[qi].append(doc_idx)

            retrieval_rounds_per_item[qi].append(
                {
                    "round_idx": round_idx,
                    "conditioning_evidence": evidence_records_per_question[qi],
                    "conditioning_summary": (
                        evidence_records_per_question[qi][0]["snippet"]
                        if evidence_records_per_question[qi]
                        and evidence_records_per_question[qi][0].get("doc_idx") == -1
                        else ""
                    ),
                    "generated_queries": generated_queries_per_item[qi],
                    "probe_queries": probe_queries,
                    "retrieval_routes": route_records,
                    "candidate_count_after_round": len(candidate_doc_idxs_per_item[qi]),
                    "current_round_candidate_count": len(
                        current_round_doc_idxs_per_item[qi]
                    ),
                }
            )

        if round_idx < num_rounds - 1:
            current_round_summary_doc_idxs_per_item = [
                select_docs_for_summary_update(
                    doc_idxs=current_round_doc_idxs_per_item[qi],
                    doc_scores=current_round_doc_scores_per_item[qi],
                    top_k=args.summary_new_doc_topk,
                )
                for qi in range(num_queries)
            ]

            updated_evidence_records_per_question = load_or_generate_incremental_summaries(
                questions=queries,
                previous_evidence_records_per_question=evidence_records_per_question,
                current_probe_queries_per_question=round_probe_queries_per_item,
                current_round_doc_idxs_per_question=current_round_summary_doc_idxs_per_item,
                docs=docs,
                round_idx=round_idx,
                args=args,
                cache_path=query_cache,
            )

            for qi in range(num_queries):
                retrieval_rounds_per_item[qi][-1]["summary_source_doc_idxs"] = (
                    current_round_summary_doc_idxs_per_item[qi]
                )
                retrieval_rounds_per_item[qi][-1]["summary_after_round"] = (
                    updated_evidence_records_per_question[qi][0]["snippet"]
                    if updated_evidence_records_per_question[qi]
                    else ""
                )

            evidence_records_per_question = updated_evidence_records_per_question

    all_rankings = []
    payloads = []

    for qi in range(num_queries):
        candidate_doc_idxs = candidate_doc_idxs_per_item[qi]
        probe_queries = accumulated_probe_queries[qi]
        probe_embeddings = np.asarray(
            accumulated_probe_embeddings[qi],
            dtype=np.float32,
        ).reshape(-1, doc_embeddings_p.shape[1])

        final_scores, base_scores, guarded_scores, probe_doc_scores = (
            rerank_candidates_with_bridge(
                probe_query_embs_q=probe_embeddings,
                candidate_doc_idxs=candidate_doc_idxs,
                doc_embeddings_p=doc_embeddings_p,
                contrastive_query_embeddings_q=tensors[
                    "contrastive_query_embeddings_q"
                ],
                contrastive_query_embeddings_p=tensors[
                    "contrastive_query_embeddings_p"
                ],
                valid_query_mask=tensors["valid_query_mask"],
                tau=args.tau,
                alpha=args.alpha,
                beta=args.beta,
                facet_representation=args.facet_representation,
                no_refinement=args.ablation == "no_refinement",
            )
        )

        relevance_order = np.argsort(-final_scores, kind="stable")

        rerank_order, coverage_selection = coverage_aware_greedy_ranking(
            candidate_doc_idxs=candidate_doc_idxs,
            relevance_scores=final_scores,
            probe_doc_scores=probe_doc_scores,
            doc_embeddings_p=doc_embeddings_p,
            top_k=args.coverage_topk,
            lambda_cov=0.0 if args.ablation == "no_coverage" else args.lambda_cov,
            lambda_red=0.0 if args.ablation == "no_coverage" else args.lambda_red,
        )

        ranked_doc_idxs = [candidate_doc_idxs[idx] for idx in rerank_order]
        all_rankings.append(ranked_doc_idxs[:max_k])

        payload_topk = min(args.payload_score_topk, len(rerank_order))
        payload_offsets = rerank_order[:payload_topk]

        payloads.append(
            {
                "num_rounds": num_rounds,
                "generated_queries": [
                    query
                    for round_record in retrieval_rounds_per_item[qi]
                    for query in round_record["generated_queries"]
                ],
                "probe_queries": probe_queries,
                "candidate_count": len(candidate_doc_idxs),
                "retrieval_rounds": retrieval_rounds_per_item[qi],
                "initial_rankings": [
                    {
                        "round_idx": round_record["round_idx"],
                        **route,
                    }
                    for round_record in retrieval_rounds_per_item[qi]
                    for route in round_record["retrieval_routes"]
                ],
                "coverage_score_source": (
                    "base_probe_scores"
                    if args.ablation == "no_refinement"
                    else "refined_probe_scores"
                ),
                "coverage_selection": coverage_selection,
                "relevance_ranking_top": [
                    int(candidate_doc_idxs[idx])
                    for idx in relevance_order[: args.payload_score_topk]
                ],
                "contrastive_scores": [
                    {
                        "doc_idx": int(candidate_doc_idxs[idx]),
                        "base_score": float(base_scores[idx]),
                        "guarded_score": float(guarded_scores[idx]),
                        "final_score": float(final_scores[idx]),
                    }
                    for idx in payload_offsets
                ],
            }
        )

        if qi == 0 or (qi + 1) % 50 == 0:
            print(f"[Final Rerank] processed {qi + 1}/{len(queries)} queries")

    return all_rankings, payloads


def load_contrastive_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run python -m contrastive_retrieval.index first."
        )
    corpus = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("The corpus index must be a non-empty JSON list")
    fields = (
        "doc_embeddings_p",
        "contrastive_query_embeddings_q",
        "contrastive_query_embeddings_p",
        "valid_query_mask",
    )
    for index, item in enumerate(corpus):
        missing = set(fields) - item.keys()
        if missing:
            raise ValueError(f"Index document {index} is missing: {sorted(missing)}")
    docs = [
        normalize_doc(
            item.get("title", ""),
            item.get("text", item.get("paragraph_text", item.get("body", ""))),
        )
        for item in corpus
    ]
    tensors = {
        key: np.asarray(
            [item[key] for item in corpus],
            dtype=bool if key == "valid_query_mask" else np.float32,
        )
        for key in fields
    }
    passage, facet_q, facet_p, mask = (tensors[key] for key in fields)
    if passage.ndim != 2 or passage.shape[1] == 0:
        raise ValueError(
            "Document embeddings must have shape (documents, embedding_dimension)"
        )
    if facet_q.ndim != 3 or facet_q.shape != facet_p.shape:
        raise ValueError(
            "Query and passage facet tensors must have identical 3D shapes"
        )
    if (
        facet_q.shape[0] != len(docs)
        or facet_q.shape[2] != passage.shape[1]
        or mask.shape != facet_q.shape[:2]
    ):
        raise ValueError("Document, facet, and mask tensor dimensions disagree")
    for name in fields[:-1]:
        if not np.isfinite(tensors[name]).all():
            raise ValueError(f"Non-finite embedding values in {name}")
    if not np.allclose(np.linalg.norm(passage, axis=1), 1.0, atol=1e-3):
        raise ValueError("Passage embeddings must be L2-normalized; rebuild the index")
    return docs, tensors


def validate_index_metadata(index_path, args):
    metadata_path = index_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        warnings.warn(
            "Legacy index has no metadata sidecar; encoder compatibility cannot be verified. "
            "Rebuild it with contrastive_retrieval.index for a reproducible experiment.",
            stacklevel=2,
        )
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata.get("embedding_model", metadata.get("dense_model"))
    if model != args.dense_model:
        raise ValueError(
            f"Index encoder {model!r} differs from --dense_model {args.dense_model!r}"
        )
    if metadata.get("encoder_max_seq_length") != args.encoder_max_seq_length:
        raise ValueError("Index and query encoder_max_seq_length differ")
    if metadata.get("prefixes") != {"query": "query: ", "passage": "passage: "}:
        raise ValueError("Index metadata must use E5 query: and passage: prefixes")
    if metadata.get("normalized_embeddings") is not True:
        raise ValueError("Index metadata does not declare normalized embeddings")
    expected = metadata.get("index_sha256")
    actual = getattr(args, "index_sha256", None) or file_sha256(index_path)
    if expected and expected != actual:
        raise ValueError(
            "Index checksum differs from its metadata; rebuild or restore the matching index"
        )
    return metadata


def load_samples(dataset_dir, dataset, max_queries):
    path = dataset_dir / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Question samples not found: {path}")
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset samples must be a non-empty JSON list")
    if max_queries > 0:
        samples = samples[:max_queries]
    if any(
        not isinstance(sample.get("question"), str) or not sample["question"].strip()
        for sample in samples
    ):
        raise ValueError("Each dataset sample must have a non-empty question string")
    return [sample["question"] for sample in samples], samples


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--corpus_index", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--query_cache", type=Path)
    parser.add_argument(
        "--max_queries", type=int, default=0, help="0 evaluates all prepared samples"
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--num_rounds", type=int, default=4)
    parser.add_argument(
        "--candidate_topk",
        type=int,
        default=10,
        help="Passages per probe; independent of --ks",
    )
    parser.add_argument(
        "--ablation",
        choices=["none", "no_coverage", "no_refinement", "no_exploration"],
        default="none",
    )
    parser.add_argument(
        "--facet_representation", choices=["dual", "query"], default="dual"
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--beta", type=float, default=0.5, help="BASE relevance weight in Eq. (6)"
    )
    parser.add_argument("--lambda_cov", type=float, default=1.0)
    parser.add_argument(
        "--lambda_red",
        type=float,
        default=0.0,
        help="Optional redundancy extension; paper uses 0",
    )
    parser.add_argument(
        "--coverage_topk", type=int, help="Greedy prefix length; defaults to --topk"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense_model", default="intfloat/e5-large-v2")
    parser.add_argument("--encoder_device", default="cuda")
    parser.add_argument("--encoder_batch_size", type=int, default=128)
    parser.add_argument("--encoder_max_seq_length", type=int, default=512)
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_batch_size", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_top_p", type=float, default=1.0)
    parser.add_argument("--llm_max_tokens", type=int, default=512)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--skip_query_expansion",
        action="store_true",
        help="Diagnostic: use only the original question",
    )
    parser.add_argument("--summary_batch_size", type=int, default=64)
    parser.add_argument("--summary_temperature", type=float, default=0.0)
    parser.add_argument("--summary_top_p", type=float, default=1.0)
    parser.add_argument("--summary_max_tokens", type=int, default=512)
    parser.add_argument("--summary_max_words", type=int, default=400)
    parser.add_argument(
        "--summary_source_snippet_chars",
        type=int,
        default=320,
        help="0 includes each full source passage",
    )
    parser.add_argument(
        "--summary_new_doc_topk",
        type=int,
        default=20,
        help="0 includes all current-round passages",
    )
    parser.add_argument("--payload_initial_topk", type=int, default=10)
    parser.add_argument("--payload_score_topk", type=int, default=10)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.coverage_topk is None:
        args.coverage_topk = args.topk
    positive = (
        "topk",
        "num_rounds",
        "candidate_topk",
        "encoder_batch_size",
        "encoder_max_seq_length",
        "generation_batch_size",
        "tensor_parallel_size",
        "llm_max_tokens",
        "summary_batch_size",
        "summary_max_tokens",
        "summary_max_words",
    )
    if any(getattr(args, name) <= 0 for name in positive) or any(
        k <= 0 for k in args.ks
    ):
        parser.error(
            "Retrieval cutoffs, round counts, sequence lengths, and batch sizes must be positive"
        )
    nonnegative = (
        "max_queries",
        "coverage_topk",
        "summary_source_snippet_chars",
        "summary_new_doc_topk",
        "seed",
        "payload_initial_topk",
        "payload_score_topk",
        "alpha",
        "lambda_cov",
        "lambda_red",
    )
    if any(getattr(args, name) < 0 for name in nonnegative):
        parser.error(
            "Counts, alpha, and coverage/redundancy weights must be non-negative"
        )
    if not 0 <= args.beta <= 1 or args.tau <= 0:
        parser.error("beta must be in [0, 1], and tau must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1]")
    for prefix in ("llm", "summary"):
        if (
            getattr(args, f"{prefix}_temperature") < 0
            or not 0 < getattr(args, f"{prefix}_top_p") <= 1
        ):
            parser.error(
                "Generation temperatures must be non-negative and top_p in (0, 1]"
            )
    return args


def main(argv=None):
    args = parse_args(argv)
    multiprocessing.set_start_method("spawn", force=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.corpus_index = (
        (
            args.corpus_index
            or REPO_ROOT / "artifacts" / "indexes" / f"{args.dataset}.json"
        )
        .expanduser()
        .resolve()
    )
    args.out = (
        (args.out or REPO_ROOT / "outputs" / "retrieval" / f"{args.dataset}_ours.json")
        .expanduser()
        .resolve()
    )
    args.query_cache = (
        (
            args.query_cache
            or REPO_ROOT / "artifacts" / "cache" / f"{args.dataset}_online.json"
        )
        .expanduser()
        .resolve()
    )
    queries, samples = load_samples(args.dataset_dir, args.dataset, args.max_queries)
    args.index_sha256 = file_sha256(args.corpus_index)
    args.dataset_sha256 = file_sha256(args.dataset_dir / f"{args.dataset}.json")
    metadata = validate_index_metadata(args.corpus_index, args)
    docs, tensors = load_contrastive_index(args.corpus_index)
    if (
        metadata
        and metadata.get("embedding_dim") != tensors["doc_embeddings_p"].shape[1]
    ):
        raise ValueError("Index embedding dimension differs from metadata")
    text_map, title_map = build_doc_index_maps(docs)
    for sample in samples:
        extract_gold_indices(sample, text_map, title_map)
    args.ks = sorted(set(args.ks + [args.topk]))
    rankings, payloads = run_retrieval(
        docs, queries, tensors, args, max(args.ks), args.query_cache
    )
    results = evaluate_results(
        docs, queries, samples, rankings, payloads, args.ks, args.topk
    )
    results["config"] = {
        **vars(args),
        "effective_num_rounds": (
            1 if args.ablation == "no_exploration" else args.num_rounds
        ),
        "prompt_sha256": stable_hash(
            [QUERY_EXPANSION_PROMPT, CANDIDATE_SUMMARY_PROMPT]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved results to {args.out}")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
