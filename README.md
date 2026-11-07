# contrastive-evidence-exploration


```text
contrastive_retrieval/
  __init__.py
  data.py                  # dataset preparation and validation
  gold.py                  # supporting-passage matching
  index.py                 # neighbor retrieval, facet generation, and indexing
  retrieve.py              # exploration, refinement, coverage, and retrieval metrics
  qa.py                    # reader inference and EM/F1 evaluation
  report.py                # conversion of saved JSON results to CSV
scripts/run.py             # launcher for indexing, retrieval, ablations, and QA
requirements.txt
README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Data

To prepare JSON/JSONL inputs and validate supporting passages:

```bash
python -m contrastive_retrieval.data prepare \
  --dataset [musique/hotpotqa/2wikimultihopqa] \
  --samples_file "<samples_file_path>" \
  --corpus_file "<corpus_file_path>" \
  --output_dir "<dataset_dir>"

python -m contrastive_retrieval.data validate \
  --dataset [musique/hotpotqa/2wikimultihopqa] \
  --dataset_dir "<dataset_dir>"
```

## Indexing, retrieval, and QA

```bash
# 1. Generate contrastive facets and build the index.
python -m contrastive_retrieval.index --dataset [musique/hotpotqa/2wikimultihopqa]

# 2. Retrieve passages and calculate retrieval metrics.
python -m contrastive_retrieval.retrieve --dataset [musique/hotpotqa/2wikimultihopqa]

# 3. Generate answers and calculate EM/F1.
python -m contrastive_retrieval.qa \
  --dataset [musique/hotpotqa/2wikimultihopqa] \
  --retrieval "<retrieved_file_path>" \
# --predictions_file "<predictions_file_path>" \
  --topk_docs 5 \
  --out "<qa_output_file_path>"
```

Indexing writes `artifacts/indexes/<dataset>.json`, a matching `.meta.json`,
and a resumable `.queries.json` checkpoint. Preserve them together. Rerunning
resumes compatible facet checkpoints; changed inputs/settings require separate
cache/output paths. Online generation caches are under `artifacts/cache/`.

## Experiment recipes

The launcher resolves relative paths from this directory. It defaults to all three
benchmarks; specify `--datasets` to restrict a run:

```bash
python scripts/run.py --stage index --datasets [musique/hotpotqa/2wikimultihopqa]
python scripts/run.py --stage retrieve --datasets [musique/hotpotqa/2wikimultihopqa]
python scripts/run.py --stage qa --datasets [musique/hotpotqa/2wikimultihopqa]
```

```bash
python -m contrastive_retrieval.report \
  "<retrieved_file_path>" \
  --percent \
  --out "<retrieval_summary_file_path>"

python -m contrastive_retrieval.report \
  "<qa_output_file_path>" \
  --percent \
  --out "<qa_summary_file_path>"
```

