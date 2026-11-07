import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ["none", "no_coverage", "no_refinement", "no_exploration"]


def ours_tag(variant):
    return "ours" if variant == "none" else f"ours_{variant}"


def build_commands(args):
    commands = []
    for dataset in args.datasets:
        index = args.index_dir / f"{dataset}.json"
        retrieval_dir = args.output_dir / "retrieval"
        base = ["--dataset", dataset, "--dataset_dir", str(args.dataset_dir), "--seed", str(args.seed)]
        if args.stage == "index":
            commands.append([sys.executable, "-m", "contrastive_retrieval.index", *base,
                             "--output_file", str(index), "--encoder_device", args.encoder_device])
        elif args.stage in {"retrieve", "ablation"}:
            variants = args.ablations or (ABLATIONS if args.stage == "ablation" else ["none"])
            for variant in variants:
                commands.append([sys.executable, "-m", "contrastive_retrieval.retrieve", *base,
                                 "--corpus_index", str(index), "--ablation", variant,
                                 "--encoder_device", args.encoder_device,
                                 "--out", str(retrieval_dir / f"{dataset}_{ours_tag(variant)}.json")])
        elif args.stage == "qa":
            for variant in args.ablations or ["none"]:
                tag = ours_tag(variant)
                commands.append([sys.executable, "-m", "contrastive_retrieval.qa", *base,
                                 "--retrieval", str(retrieval_dir / f"{dataset}_{tag}.json"),
                                 "--topk_docs", "5", "--out", str(args.output_dir / "qa" / f"{dataset}_{tag}.json")])
        else:
            raise ValueError(f"Unknown stage: {args.stage}")
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["index", "retrieve", "ablation", "qa"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["musique", "hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--dataset_dir", type=Path, default=Path("data"))
    parser.add_argument("--index_dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS)
    parser.add_argument("--encoder_device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    for field in ("dataset_dir", "index_dir", "output_dir"):
        path = getattr(args, field)
        setattr(args, field, path if path.is_absolute() else ROOT / path)
    try:
        commands = build_commands(args)
    except ValueError as exc:
        parser.error(str(exc))
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
