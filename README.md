# Not All OCR Errors Are Equal: Measuring Historical Information Loss

CER counts how many characters OCR got wrong, not what those errors cost. A typo in "the" and a typo in a person's name weigh the same under CER, but only one of them hides the person from a search. We score each OCR error by the damage it does to named entity recognition, entity linking and retrieval on historical newspapers, and test the scores by repairing the most damaging errors first.

| Part | Folder |
|---|---|
| Benchmark: NER and EL annotation, consensus, GT to OCR projection | `src/benchmark/` |
| Named entity recognition | `src/ner/` |
| Entity linking | `src/el/` |
| Retrieval | `src/retrieval/` |

The final annotations are released at [huggingface.co/datasets/trung11/OCR-Information-Loss](https://huggingface.co/datasets/trung11/OCR-Information-Loss).

---

## Installation

Python 3.11.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

Some dependencies are only needed for one step:

- `vllm`: the Llama annotators.
- `flair`: the hmBERT robustness check.
- `genalog`: the GT to OCR projection.

---

## Repository Overview

- `config/prompts.yaml`: every LLM prompt and the GLiNER labels.
- `config/params.yaml`: models and hyperparameters.
- `src/config.py`: loads the two config files.
- `src/data.py`: loads HIPE-OCRepair records, and the `--shard` and `--rho` flags.
- `src/spans.py`: word-level edit spans, corruption, repair and RepairAUC.
- `src/local_damage.py`: local damage and the CER predictors.
- `src/merge_shards.py`: merges the outputs of `--shard` runs.
- `src/evaluate.py`: all results.
- `src/benchmark/`: NER and EL annotators, consensus, agreement and GT to OCR projection.
- `src/ner/`: tagger, predictions, outcomes, LIME and occlusion attribution, and repair.
- `src/el/`: linker, predictions, outcomes, LIME and occlusion attribution, and repair.
- `src/retrieval/`: units, queries, BM25, damage and repair.

---

## Data Preparation

### HIPE-OCRepair-2026

```bash
git clone https://github.com/hipe-eval/HIPE-OCRepair-2026-data.git data/raw   # v0.9, commit 1bfecaf
```

### Entity linking resources

- The EL scripts need `lang_title2wikidataID-normalized_with_redirect.pkl` from [GENRE](https://github.com/facebookresearch/GENRE) in the repository root.
- Candidate generation needs [entity-fishing](https://github.com/kermitt2/entity-fishing) running at `localhost:8090`.

---

## Running the Pipeline

Every script runs from the repository root.

#### Main options

- `--test`: selects the test split.
- `--shard i/n`: splits a script across GPUs. Merge the outputs with `python -m src.merge_shards <name>`.
- `--rho`: leaves residual character errors in each repair, for the imperfect repair experiment.

### Step 1: NER annotation

An entity is kept when two of the three annotators agree.

```bash
OPENAI_API_KEY=... python -m src.benchmark.ner_gpt --test
python -m src.benchmark.ner_llama --test
python -m src.benchmark.ner_gliner --test
python -m src.benchmark.consensus_ner --test
python -m src.benchmark.iaa --test
python -m src.benchmark.project --test
```

### Step 2: Entity linking annotation

```bash
WIKIDATA_CONTACT=... python -m src.benchmark.candidates --test
OPENAI_API_KEY=... python -m src.benchmark.el_gpt --test
python -m src.benchmark.el_llama --test
python -m src.benchmark.consensus_el --test
```

### Step 3: NER

```bash
python -m src.ner.tag --test
python -m src.ner.tag --test --ocr
python -m src.ner.outcomes --test
python -m src.ner.lime --test
python -m src.ner.occlusion
python -m src.ner.repair --test                # also --rho=0.1 and --rho=0.25
```

### Step 4: Entity linking

```bash
python -m src.el.tag --test
python -m src.el.outcomes --test
python -m src.el.lime --test
python -m src.el.occlusion
python -m src.el.repair --test                 # also --rho=0.1 and --rho=0.25
```

### Step 5: Local damage for NER and EL

```bash
python -m src.local_damage --test
```

### Step 6: Retrieval

```bash
python -m src.retrieval.units --test
python -m src.retrieval.queries --test
python -m src.retrieval.bm25 --test
python -m src.retrieval.metrics --test
python -m src.retrieval.damage --test
python -m src.retrieval.repair --test          # also --rho=0.1 and --rho=0.25
```

### Step 7: Results

```bash
python -m src.evaluate
```

The results are written to `data/results/evaluation_test.json`, with the human validation decisions from `data/annotations/annot_{lang}.jsonl` applied. Confidence intervals use 2,000 document-level bootstrap resamples.

### hmBERT robustness check

`src/ner/hmbert.py` tags GT and OCR. Step 3 then runs with hmBERT in place of the tagger in `src/ner/tagger.py`, writing to `data/results/hmbert/`.
