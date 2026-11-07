import argparse
import hashlib
import json
from pathlib import Path

from .gold import build_doc_index_maps, extract_gold_indices


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or not all(isinstance(x, dict) for x in records):
        raise ValueError(f"{path}: expected a nonempty list of objects (JSON or JSONL).")
    return records


def text_value(value):
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value or "")


def normalize_doc(title, text):
    return f"{title or ''}\n{text_value(text)}"


def sample_passages(sample, supporting_only=False):
    if "supporting_facts" in sample:
        titles = {fact[0] for fact in sample["supporting_facts"]}
        for title, text in sample.get("context", []):
            if not supporting_only or title in titles:
                yield {"title": str(title), "text": text_value(text), "is_supporting": title in titles}
    else:
        entries = sample.get("contexts", sample.get("paragraphs", []))
        for item in entries:
            if "is_supporting" not in item:
                raise ValueError("Each contexts/paragraphs entry must explicitly declare is_supporting.")
            if not isinstance(item["is_supporting"], bool):
                raise ValueError("is_supporting must be a JSON boolean.")
            if not supporting_only or item["is_supporting"]:
                yield {"title": str(item.get("title", "")),
                       "text": text_value(item.get("text", item.get("paragraph_text", ""))),
                       "is_supporting": item["is_supporting"]}


def answer_aliases(sample):
    raw = sample.get("answer", sample.get("gold_ans", sample.get("reference")))
    values = []
    if raw is not None:
        values.extend(raw if isinstance(raw, list) else [raw])
    elif "obj" in sample:
        for key in ("obj", "possible_answers", "o_wiki_title", "o_aliases"):
            value = sample.get(key)
            if value is not None:
                values.extend(value if isinstance(value, list) else [value])
    extra = sample.get("answer_aliases", [])
    values.extend(extra if isinstance(extra, list) else [extra])
    answers = {str(value) for value in values if value is not None and str(value).strip()}
    if not answers:
        raise ValueError(f"Question {sample.get('id', sample.get('_id', '?'))} has no answer.")
    return answers


def validate_records(corpus, samples):
    if not corpus or not samples:
        raise ValueError("Corpus and questions must both be nonempty.")
    documents, docs = set(), []
    for i, doc in enumerate(corpus):
        if not isinstance(doc.get("title"), str) or not text_value(doc.get("text", doc.get("paragraph_text"))):
            raise ValueError(f"Corpus row {i} requires a string title and nonempty text/paragraph_text.")
        text = normalize_doc(doc["title"], doc.get("text", doc.get("paragraph_text")))
        documents.add(text)
        docs.append(text)
    text_map, title_map = build_doc_index_maps(docs)
    seen_ids, whitespace_count = set(), 0
    for i, sample in enumerate(samples):
        if not isinstance(sample.get("question"), str) or not sample["question"].strip():
            raise ValueError(f"Question row {i} has an empty/missing question.")
        identifier = sample.get("id", sample.get("_id"))
        if identifier is not None:
            if str(identifier) in seen_ids:
                raise ValueError(f"Duplicate question id: {identifier}")
            seen_ids.add(str(identifier))
        answer_aliases(sample)
        gold = list(sample_passages(sample, supporting_only=True))
        extract_gold_indices(sample, text_map, title_map)
        for doc in gold:
            if normalize_doc(doc["title"], doc["text"]) not in documents:
                whitespace_count += 1
    return {"corpus_size": len(corpus), "question_count": len(samples),
            "unique_passages": len(documents), "gold_passages_using_title_fallback": 0,
            "gold_passages_using_whitespace_normalization": whitespace_count}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args):
    samples = read_records(args.samples_file)
    if args.corpus_file:
        corpus = read_records(args.corpus_file)
        corpus = [{**doc, "text": text_value(doc.get("text", doc.get("paragraph_text", "")))} for doc in corpus]
        source_kind = "provided_corpus"
    else:
        corpus, seen = [], set()
        for sample in samples:
            for doc in sample_passages(sample):
                key = normalize_doc(doc["title"], doc["text"])
                if key not in seen:
                    seen.add(key)
                    corpus.append({"title": doc["title"], "text": doc["text"]})
        source_kind = "sample_context_union"
    summary = validate_records(corpus, samples)
    output = Path(args.output_dir)
    targets = [output / f"{args.dataset}.json", output / f"{args.dataset}_corpus.json",
               output / f"{args.dataset}.provenance.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise ValueError("Prepared output exists; choose another output directory or pass --overwrite.")
    write_json(targets[0], samples)
    write_json(targets[1], corpus)
    sources = [Path(args.samples_file)] + ([Path(args.corpus_file)] if args.corpus_file else [])
    write_json(targets[2], {"dataset": args.dataset, "corpus_source": source_kind,
                           "sources": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
                           **summary})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--samples_file", type=Path, required=True)
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus_file", type=Path)
    source.add_argument("--derive_corpus", action="store_true")
    prep.add_argument("--output_dir", type=Path, default=Path("data"))
    prep.add_argument("--overwrite", action="store_true")
    check = sub.add_parser("validate")
    check.add_argument("--dataset", required=True)
    check.add_argument("--dataset_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not args.dataset or Path(args.dataset).name != args.dataset or args.dataset in {".", ".."}:
        parser.error("--dataset must be a file prefix")
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            print(json.dumps(validate_records(read_records(args.dataset_dir / f"{args.dataset}_corpus.json"),
                                               read_records(args.dataset_dir / f"{args.dataset}.json")), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
