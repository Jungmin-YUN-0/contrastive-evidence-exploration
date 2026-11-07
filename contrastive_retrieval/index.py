from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable

import numpy as np


PROMPT = """Write queries that are answerable from the target passage but not from the reference passage.

Rules:
- Generate 2 to 4 queries.
- Each query must be answerable using only the target passage.
- Each query must not be answerable from the reference passage.
- Each query should mention explicit entities and concrete relations whenever possible.
- Avoid ambiguous pronouns and generic topic queries.
- Return JSON only, without explanation.

Target passage:
{target_passage}

Reference passage:
{reference_passage}

Return format:
{{"queries": ["...", "..."]}}
"""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def document_text(doc: dict[str, Any]) -> str:
    text = doc.get("text", doc.get("paragraph_text", doc.get("body", "")))
    if isinstance(text, list):
        if not all(isinstance(part, str) for part in text):
            raise ValueError("Document text lists must contain strings.")
        text = " ".join(text)
    if not isinstance(text, str):
        raise ValueError("Document text must be a string or a list of strings.")
    return text.strip()


def format_passage(doc: dict[str, Any], *, prompt: bool = False) -> str:
    title = doc.get("title", "")
    if not isinstance(title, str):
        raise ValueError("Document titles must be strings.")
    title, text = title.strip(), document_text(doc)
    if prompt and title:
        return f"Title: {title}\nText: {text}"
    return f"{title}\n{text}".strip() if title else text


def passage_identity(doc: dict[str, Any]) -> str:
    text = doc.get("text", doc.get("paragraph_text", doc.get("body", "")))
    if isinstance(text, list):
        text = " ".join(text)
    return f"{doc.get('title', '')}\n{text}"


def load_corpus(path: Path, max_docs: int = 0) -> list[dict[str, Any]]:
    if max_docs < 0:
        raise ValueError("max_docs must be non-negative.")
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a non-empty JSON list of document objects.")
    if max_docs:
        corpus = corpus[:max_docs]
    for position, doc in enumerate(corpus):
        if not isinstance(doc, dict) or not format_passage(doc):
            raise ValueError(f"Document {position} is not a non-empty passage object.")
    return corpus


def deduplicate(queries: list[str]) -> list[str]:
    unique, seen = [], set()
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each facet query must be a non-empty string.")
        query = query.strip()
        key = " ".join(query.casefold().split())
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def parse_queries(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Facet generation must return JSON with a queries list.") from exc
    if isinstance(value, dict):
        value = value.get("queries")
    if not isinstance(value, list):
        raise ValueError("Facet queries must be a list, or an object with a queries list.")
    return deduplicate(value)


def precomputed_facets(corpus: list[dict[str, Any]]) -> list[list[str]]:
    result = []
    for position, doc in enumerate(corpus):
        if "const_query_by_reference" in doc:
            rows = doc["const_query_by_reference"]
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError(f"Document {position} has invalid per-reference queries.")
            queries = deduplicate([q for row in rows for q in parse_queries(row.get("queries"))])
        else:
            key = next((key for key in ("const_query", "all_queries_by_doc") if key in doc), None)
            if key is None:
                if "kept_queries_by_doc" in doc:
                    raise ValueError(f"Document {position} only has legacy kept_queries_by_doc;")
                raise ValueError(f"Document {position} has no precomputed facet queries.")
            queries = parse_queries(doc[key])
        result.append(queries)
    if not any(result):
        raise ValueError("No precomputed facets found;")
    return result


def encode(texts: list[str], model: Any, *, prefix: str, batch_size: int) -> np.ndarray:
    embeddings = np.asarray(model.encode(
        [f"{prefix}: {text}" for text in texts],
        batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
        raise ValueError("Encoder returned an invalid embedding matrix.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.all(np.isfinite(embeddings)) or np.any(norms <= 0):
        raise ValueError("Encoder returned non-finite or zero embeddings.")
    return embeddings / norms


def nearest_neighbors(embeddings: np.ndarray, k: int = 10, batch_size: int = 256) -> list[list[int]]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not len(embeddings) or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have shape (number of documents, dimension).")
    if k <= 0 or batch_size <= 0:
        raise ValueError("k and batch_size must be positive.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms <= 0):
        raise ValueError("Neighbor embeddings must be finite and nonzero.")
    embeddings = embeddings / norms
    k = min(k, len(embeddings) - 1)
    neighbors = []
    for start in range(0, len(embeddings), batch_size):
        scores = embeddings[start:start + batch_size] @ embeddings.T
        scores[np.arange(len(scores)), np.arange(start, start + len(scores))] = -np.inf
        # Stable sorting gives a reproducible document-position tie break.
        neighbors.extend(np.argsort(-scores, axis=1, kind="stable")[:, :k].tolist())
    return neighbors


def corpus_fingerprint(corpus: list[dict[str, Any]]) -> str:
    payload = json.dumps([
        {"title": doc.get("title", "").strip(), "text": document_text(doc)}
        for doc in corpus
    ], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_facets(
    corpus: list[dict[str, Any]], neighbors: list[list[int]],
    generator: Callable[[list[str], list[int]], list[str]] | None,
    cache_path: Path, config: dict[str, Any], batch_size: int = 32,
    max_generation_attempts: int = 3,
) -> list[list[str]]:

    if batch_size <= 0 or len(neighbors) != len(corpus):
        raise ValueError("Invalid generation batch size or neighbor row count.")
    if max_generation_attempts <= 0:
        raise ValueError("max_generation_attempts must be positive.")
    pairs = []
    for target, references in enumerate(neighbors):
        if not references or len(set(references)) != len(references):
            raise ValueError(f"Document {target} needs distinct nonself references.")
        for reference in references:
            if not isinstance(reference, int) or reference == target or not 0 <= reference < len(corpus):
                raise ValueError(f"Invalid reference {reference} for document {target}.")
            pairs.append((target, reference))
    signature = {
        "corpus_sha256": corpus_fingerprint(corpus), "neighbors": neighbors,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), **config,
    }
    cache = {"schema_version": 1, "configuration": signature, "pairs": {}}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("schema_version") != 1 or cache.get("configuration") != signature:
            raise ValueError(f"Query cache does not match this corpus/configuration: {cache_path}.")
        if not isinstance(cache.get("pairs"), dict):
            raise ValueError("Query cache pairs must be an object.")
    completed = cache["pairs"]
    attempts = cache.setdefault("attempts", {})
    failures = cache.setdefault("failures", {})
    if (not isinstance(attempts, dict) or not isinstance(failures, dict)
            or any(type(count) is not int or count < 0 for count in attempts.values())):
        raise ValueError("Query cache attempt counts/failures are invalid.")
    cache["max_generation_attempts"] = max_generation_attempts
    missing = []
    for target, reference in pairs:
        key = f"{target}:{reference}"
        if key not in completed:
            missing.append((target, reference))
        elif not isinstance(completed[key], dict) or not 2 <= len(parse_queries(completed[key].get("queries"))) <= 4:
            raise ValueError(f"Invalid cached facets for pair {key}.")
    if missing and generator is None:
        raise ValueError(f"Query cache is incomplete: {len(missing)} target/reference pairs remain.")
    while missing:
        eligible = [pair for pair in missing if attempts.get(f"{pair[0]}:{pair[1]}", 0) < max_generation_attempts]
        if not eligible:
            exhausted = ", ".join(f"{target}:{reference}" for target, reference in missing[:10])
            raise ValueError("Invalid generation")
        for start in range(0, len(eligible), batch_size):
            batch = eligible[start:start + batch_size]
            prompts, seeds = [], []
            for target, reference in batch:
                key = f"{target}:{reference}"
                attempt = attempts.get(key, 0) + 1
                attempts[key] = attempt
                prompts.append(PROMPT.format(
                    target_passage=format_passage(corpus[target], prompt=True),
                    reference_passage=format_passage(corpus[reference], prompt=True),
                ))
                seeds.append((int(config.get("seed", 42)) + target * len(corpus) + reference
                              + (attempt - 1) * len(corpus) ** 2) % (2**32))
            atomic_json(cache_path, cache)
            outputs = generator(prompts, seeds)
            if len(outputs) != len(batch):
                raise ValueError("Generator returned a different number of outputs than prompts.")
            for (target, reference), raw in zip(batch, outputs):
                key = f"{target}:{reference}"
                try:
                    queries = parse_queries(raw)
                    if not 2 <= len(queries) <= 4:
                        raise ValueError("Expected 2–4 distinct facet queries.")
                    completed[key] = {"queries": queries, "raw_output": raw,
                                      "generation_attempt": attempts[key]}
                    failures.pop(key, None)
                except ValueError as exc:
                    failures[key] = {"attempt": attempts[key], "error": str(exc), "raw_output": raw}
            atomic_json(cache_path, cache)
            print(f"Facets: {len(completed)}/{len(pairs)} target/reference pairs cached.", flush=True)
        missing = [pair for pair in missing if f"{pair[0]}:{pair[1]}" not in completed]
    return [deduplicate([
        query for reference in references
        for query in completed[f"{target}:{reference}"]["queries"]
    ]) for target, references in enumerate(neighbors)]


def build_index(
    corpus: list[dict[str, Any]], facets: list[list[str]], model: Any,
    *, batch_size: int = 64, doc_embeddings: np.ndarray | None = None,
    neighbors: list[list[int]] | None = None,
) -> list[dict[str, Any]]:

    if not corpus or len(corpus) != len(facets) or batch_size <= 0:
        raise ValueError("Nonempty corpus, one facet list per document, and positive batch_size are required.")
    facets = [deduplicate(row) for row in facets]
    if doc_embeddings is None:
        doc_embeddings = encode([format_passage(doc) for doc in corpus], model, prefix="passage", batch_size=batch_size)
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    if (doc_embeddings.ndim != 2 or doc_embeddings.shape[0] != len(corpus)
            or doc_embeddings.shape[1] == 0 or not np.isfinite(doc_embeddings).all()
            or np.any(np.linalg.norm(doc_embeddings, axis=1) <= 0)):
        raise ValueError("Invalid document embedding matrix.")
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    if neighbors is not None and len(neighbors) != len(corpus):
        raise ValueError("Neighbor row count must match the corpus.")
    dimension = doc_embeddings.shape[1]
    flat = [query for row in facets for query in row]
    query_embs = encode(flat, model, prefix="query", batch_size=batch_size) if flat else np.empty((0, dimension), dtype=np.float32)
    passage_embs = encode(flat, model, prefix="passage", batch_size=batch_size) if flat else np.empty((0, dimension), dtype=np.float32)
    if query_embs.shape[1] != dimension or passage_embs.shape[1] != dimension:
        raise ValueError("Document and facet embedding dimensions differ.")
    width = max(1, max(map(len, facets)))
    output, cursor = [], 0
    for position, (doc, queries) in enumerate(zip(corpus, facets)):
        count = len(queries)
        query_rows = np.zeros((width, dimension), dtype=np.float32)
        passage_rows = np.zeros_like(query_rows)
        query_rows[:count] = query_embs[cursor:cursor + count]
        passage_rows[:count] = passage_embs[cursor:cursor + count]
        record = dict(doc)
        record.update(
            idx=doc.get("idx", position), passage_texts=passage_identity(doc),
            const_query={"queries": queries}, all_queries_by_doc=queries,
            kept_queries_by_doc=queries, doc_embeddings_p=doc_embeddings[position].tolist(),
            contrastive_query_embeddings_q=query_rows.tolist(),
            contrastive_query_embeddings_p=passage_rows.tolist(),
            valid_query_mask=[True] * count + [False] * (width - count),
        )
        record.pop("kept_margins_by_doc", None)
        if neighbors is not None:
            record["neighboring_passage"] = neighbors[position]
        output.append(record)
        cursor += count
    return output


def _load_encoder(args: argparse.Namespace) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model, device=args.encoder_device)
    model.max_seq_length = args.encoder_max_seq_length
    return model


def _generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in (
        "llm_model", "seed", "query_max_tokens", "query_temperature", "query_top_p",
        "max_model_len", "tensor_parallel_size", "llm_dtype",
    )}


def _generation_worker(corpus: list[dict[str, Any]], neighbors: list[list[int]], args: argparse.Namespace) -> None:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = llm = None

    def generate(prompts: list[str], seeds: list[int]) -> list[str]:
        nonlocal tokenizer, llm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if llm is None:
            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            llm = LLM(
                model=args.llm_model, dtype=args.llm_dtype, seed=args.seed,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
            )
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        ) for prompt in prompts]
        parameters = [SamplingParams(
            max_tokens=args.query_max_tokens, temperature=args.query_temperature,
            top_p=args.query_top_p, seed=seed,
        ) for seed in seeds]
        outputs = llm.generate(rendered, parameters, use_tqdm=True)
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]

    generate_facets(corpus, neighbors, generate, args.query_cache,
                    _generation_config(args), args.generation_batch_size,
                    args.max_generation_attempts)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. musique, 2wikimultihopqa, hotpotqa, multihoprag")
    parser.add_argument("--dataset_dir", type=Path, default=Path("data"))
    parser.add_argument("--input_file", type=Path, help="Corpus JSON path; default: <dataset_dir>/<dataset>_corpus.json")
    parser.add_argument("--output_file", type=Path, help="Index JSON path; default: artifacts/indexes/<dataset>.json")
    parser.add_argument("--query_cache", type=Path, help="Resumable pair checkpoint; default: <output stem>.queries.json")
    parser.add_argument("--skip_query_generation", action="store_true")
    parser.add_argument("--max_docs", type=int, default=0, help="Limit corpus for smoke runs; 0 means all documents")
    parser.add_argument("--embedding_model", default="intfloat/e5-large-v2")
    parser.add_argument("--encoder_device", default="cuda")
    parser.add_argument("--encoder_batch_size", type=int, default=64)
    parser.add_argument("--encoder_max_seq_length", type=int, default=512)
    parser.add_argument("--neighbor_count", type=int, default=10)
    parser.add_argument("--neighbor_search_batch_size", type=int, default=256)
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--llm_dtype", default="bfloat16", choices=("auto", "bfloat16", "float16"))
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--query_max_tokens", type=int, default=512)
    parser.add_argument("--query_temperature", type=float, default=0.7)
    parser.add_argument("--query_top_p", type=float, default=1.0)
    parser.add_argument("--generation_batch_size", type=int, default=32)
    parser.add_argument("--max_generation_attempts", type=int, default=3)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    positive = ("encoder_batch_size", "encoder_max_seq_length", "neighbor_count",
                "neighbor_search_batch_size", "tensor_parallel_size", "max_model_len",
                "query_max_tokens", "generation_batch_size", "max_generation_attempts")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Batch sizes, lengths, neighbor count, and tensor parallel size must be positive.")
    if args.max_docs < 0:
        parser.error("--max_docs must be nonnegative.")
    if not np.isfinite(args.query_temperature) or args.query_temperature < 0 or not 0 < args.query_top_p <= 1:
        parser.error("Query temperature must be finite and nonnegative; top_p must be in (0, 1].")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1].")
    if args.query_max_tokens >= args.max_model_len:
        parser.error("--query_max_tokens must be smaller than --max_model_len.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.dataset):
        parser.error("--dataset must be a simple name without directory separators.")
    input_path = (args.input_file or args.dataset_dir / f"{args.dataset}_corpus.json").expanduser().resolve()
    output_path = (args.output_file or Path("artifacts/indexes") / f"{args.dataset}.json").expanduser().resolve()
    args.query_cache = (args.query_cache or output_path.with_suffix(".queries.json")).expanduser().resolve()
    metadata_path = output_path.with_suffix(".meta.json")
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        corpus = load_corpus(input_path, args.max_docs)
        if not args.skip_query_generation and len(corpus) < 2:
            raise ValueError("At least two corpus documents are needed to generate contrastive facets.")
        facets = precomputed_facets(corpus) if args.skip_query_generation else None
        model = _load_encoder(args)
        doc_embeddings = encode([format_passage(doc) for doc in corpus], model,
                                prefix="passage", batch_size=args.encoder_batch_size)
        neighbors = nearest_neighbors(doc_embeddings, args.neighbor_count, args.neighbor_search_batch_size)
        if facets is None:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            worker = multiprocessing.get_context("spawn").Process(
                target=_generation_worker, args=(corpus, neighbors, args),
            )
            worker.start()
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"Facet generation failed (exit {worker.exitcode}); completed pairs remain in {args.query_cache}.")
            facets = generate_facets(corpus, neighbors, None, args.query_cache, _generation_config(args),
                                     max_generation_attempts=args.max_generation_attempts)
            model = _load_encoder(args)
        records = build_index(corpus, facets, model, batch_size=args.encoder_batch_size,
                              doc_embeddings=doc_embeddings, neighbors=neighbors)
        atomic_json(output_path, records)
        index_hash = hashlib.sha256()
        with output_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                index_hash.update(block)
        metadata = {
            "schema_version": 1, "dataset": args.dataset,
            "source_file": str(input_path), "corpus_sha256": corpus_fingerprint(corpus),
            "index_sha256": index_hash.hexdigest(), "num_documents": len(corpus),
            "embedding_model": args.embedding_model, "dense_model": args.embedding_model,
            "encoder_max_seq_length": args.encoder_max_seq_length,
            "embedding_dim": int(doc_embeddings.shape[1]),
            "prefixes": {"query": "query: ", "passage": "passage: "},
            "normalized_embeddings": True,
            "facet_representation": "dual_e5_query_and_passage",
            "facet_selection": "deduplicated_union_all",
            "facet_counts": [len(row) for row in facets],
            "neighbor_count": args.neighbor_count,
            "effective_neighbor_count": min(args.neighbor_count, len(corpus) - 1),
            "neighbor_text": "title_and_text",
            "precomputed_queries": args.skip_query_generation,
            "query_cache": None if args.skip_query_generation else str(args.query_cache),
            "generation": None if args.skip_query_generation else _generation_config(args),
            "max_generation_attempts": None if args.skip_query_generation else args.max_generation_attempts,
            "seed": args.seed,
        }
        atomic_json(metadata_path, metadata)
        print(f"Saved {len(records)} documents and {sum(map(len, facets))} facets to {output_path}")
        print(f"Metadata: {metadata_path}")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Indexing failed: {exc}\n")


if __name__ == "__main__":
    main()
