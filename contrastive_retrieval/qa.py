import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from .data import answer_aliases, normalize_doc, read_records, sample_passages, write_json


def normalize_answer(text):
    """Keep the supplied evaluation script's English answer normalization."""
    text = re.sub(r"[^0-9a-z ]+", " ", str(text).lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def extract_short_answer(response):
    lines = str(response).strip().splitlines()
    text = lines[0].strip() if lines else ""
    text = re.sub(r"^\s*(the\s+)?answer\b(\s+is\b)?\s*:?\s*", "", text, flags=re.I)
    return text.strip(" \"'").rstrip(".").strip()


def exact_match(prediction, gold):
    return int(any(normalize_answer(prediction) == normalize_answer(answer) for answer in gold))


def f1_score(prediction, gold):
    predicted = normalize_answer(prediction).split()
    if not predicted:
        return 0.0
    best = 0.0
    for answer in gold:
        expected = normalize_answer(answer).split()
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if overlap:
            best = max(best, 2.0 * overlap / (len(predicted) + len(expected)))
    return best


def align_samples(per_query, samples):
    if not per_query:
        raise ValueError("Retrieval results are empty.")
    ids, questions = {}, {}
    for sample in samples:
        identifier = sample.get("id", sample.get("_id"))
        if identifier is not None:
            if str(identifier) in ids:
                raise ValueError(f"Duplicate dataset id: {identifier}")
            ids[str(identifier)] = sample
        questions.setdefault(sample["question"], []).append(sample)
    same_order = len(per_query) == len(samples) and all(
        row.get("query") == sample["question"] for row, sample in zip(per_query, samples))
    matched = []
    for i, row in enumerate(per_query):
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Result row {i} requires a nonempty query.")
        identifier = row.get("id", row.get("_id", row.get("question_id")))
        if identifier is not None:
            sample = ids.get(str(identifier))
        elif same_order:
            sample = samples[i]
        else:
            candidates = questions.get(query, [])
            sample = candidates[0] if len(candidates) == 1 else None
        if sample is None or sample["question"] != query:
            raise ValueError(f"Cannot unambiguously align result row {i} with dataset: {query!r}")
        matched.append(sample)
    return matched


def gold_retrieval(samples):
    rows = []
    for sample in samples:
        docs = [normalize_doc(doc["title"], doc["text"]) for doc in sample_passages(sample, supporting_only=True)]
        if not docs:
            raise ValueError(f"No gold evidence for {sample['question']!r}")
        row = {"query": sample["question"], "top_texts": list(dict.fromkeys(docs))}
        identifier = sample.get("id", sample.get("_id"))
        if identifier is not None:
            row["id"] = identifier
        rows.append(row)
    return rows


def build_prompts(rows, topk=5):
    prompts = []
    for row in rows:
        docs = row.get("top_texts")
        if not isinstance(docs, list) or not all(isinstance(doc, str) for doc in docs):
            raise ValueError("Each retrieval row must contain a list of strings in top_texts.")
        context = "\n\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(docs[:topk]))
        content = ("Answer the question with a short answer using only the given context information. "
                   "Respond with only the answer phrase on a single line, with no explanation.\n\n"
                   f"Context:\n{context}\n\nQuestion: {row['query']}\nAnswer:")
        prompts.append([{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content}])
    return prompts


def generate(prompts, args):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    reader = LLM(model=args.model_name, tensor_parallel_size=args.num_gpus,
                 seed=args.seed, dtype="auto", gpu_memory_utilization=args.gpu_memory_utilization,
                 max_model_len=args.max_seq_len, enable_prefix_caching=True)
    tokenizer = reader.get_tokenizer()
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(tokens) + args.max_tokens > args.max_seq_len:
            raise ValueError(f"Reader prompt {i} exceeds context budget; increase --max_seq_len.")
    responses = []
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0, stop=["\n"])
    for start in range(0, len(prompts), args.batch_size):
        outputs = reader.chat(messages=prompts[start:start + args.batch_size], sampling_params=params, use_tqdm=True)
        responses.extend(item.outputs[0].text if item.outputs else "" for item in outputs)
    return responses


def evaluate(rows, samples, responses, topk=5):
    matched = align_samples(rows, samples)
    if len(responses) != len(rows):
        raise ValueError(f"Expected {len(rows)} predictions, received {len(responses)}.")
    results = []
    for row, sample, response in zip(rows, matched, responses):
        prediction = extract_short_answer(response)
        aliases = answer_aliases(sample)
        results.append({"id": sample.get("id", sample.get("_id")), "query": row["query"],
                        "pred": prediction, "raw_response": response, "gold_answers": sorted(aliases),
                        "EM": exact_match(prediction, aliases), "F1": f1_score(prediction, aliases),
                        "top_texts": row.get("top_texts", [])[:topk]})
    return {"EM": sum(row["EM"] for row in results) / len(results),
            "F1": sum(row["F1"] for row in results) / len(results),
            "num_queries": len(results), "per_query": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_dir", type=Path, default=Path("data"))
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--retrieval", type=Path)
    evidence.add_argument("--gold", action="store_true")
    parser.add_argument("--predictions_file", type=Path)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--num_gpus", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.topk_docs, args.max_tokens, args.max_seq_len, args.num_gpus, args.batch_size) <= 0:
        parser.error("Token, document, batch, and GPU counts must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1].")
    try:
        samples = read_records(args.dataset_dir / f"{args.dataset}.json")
        rows = gold_retrieval(samples) if args.gold else json.loads(args.retrieval.read_text(encoding="utf-8"))["per_query"]
        # Validate alignment and gold answers before allocating the reader model.
        for sample in align_samples(rows, samples):
            answer_aliases(sample)
        prompts = build_prompts(rows, args.topk_docs)
        if args.predictions_file:
            responses = json.loads(args.predictions_file.read_text(encoding="utf-8"))
            if not isinstance(responses, list) or not all(isinstance(x, str) for x in responses):
                raise ValueError("--predictions_file must contain a JSON list of strings in retrieval order.")
        else:
            responses = generate(prompts, args)
        results = evaluate(rows, samples, responses, args.topk_docs)
        results["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        results["config"]["evidence_source"] = "gold_oracle" if args.gold else "retrieved"
        results["config"]["reader_executed"] = args.predictions_file is None
        tag = "gold" if args.gold else args.retrieval.stem
        output = args.out or Path("outputs/qa") / f"{args.dataset}_{tag}.json"
        write_json(output, results)
        print(f"EM={results['EM']:.4f} F1={results['F1']:.4f}; saved {output}")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
