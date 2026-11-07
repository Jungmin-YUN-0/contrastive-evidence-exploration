import argparse
import csv
import json
from pathlib import Path
import sys


FIELDS = ["file", "dataset", "method", "questions", "R@5", "R@10", "MAP@10", "nDCG@10", "FSR@10", "EM", "F1"]


def result_row(path, percent=False):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not ("metrics" in result or "EM" in result):
        raise ValueError(f"{path} is not a retrieval or QA result.")
    config = result.get("config", {})
    if "ablation" in config:
        method = "ours" if config["ablation"] == "none" else f"ours_{config['ablation']}"
    else:
        method = config.get("method", Path(path).stem)
    row = {"file": str(path), "dataset": config.get("dataset", result.get("dataset", "")),
           "method": method,
           "questions": len(result.get("per_query", []))}
    metrics = result.get("metrics", {})
    for field, cutoff, name in [("R@5", "@5", "recall"), ("R@10", "@10", "recall"),
                                ("MAP@10", "@10", "MAP"), ("nDCG@10", "@10", "nDCG"),
                                ("FSR@10", "@10", "supporting_all_fraction")]:
        row[field] = metrics.get(cutoff, {}).get(name, "")
    row.update({name: result.get(name, "") for name in ("EM", "F1")})
    for field in FIELDS[4:]:
        if row[field] != "":
            row[field] = round(float(row[field]) * (100 if percent else 1), 6)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--percent", action="store_true")
    args = parser.parse_args()
    try:
        rows = [result_row(path, args.percent) for path in args.files]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stream = args.out.open("w", newline="", encoding="utf-8")
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            stream.close()


if __name__ == "__main__":
    main()
