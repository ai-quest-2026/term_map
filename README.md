# TermMap: clinical term → ICD-10-CM + SNOMED CT

TermMap maps an extracted clinical entity, together with its chunk context, to one ICD-10-CM code and one SNOMED CT concept. Every answer it gives is a code from the supplied dictionaries. It was built for the HiLabs TermMap challenge (problem statement: `TermMap A Clinical Term Mapping Engine…md`; data findings: [`docs/analysis.md`](docs/analysis.md)).

## Pipeline (v0.2, CPU only)

```
entity + chunk_text
   │  normalize: lowercase, punctuation/OCR cleanup, abbreviation expansion (ambiguous → several readings),
   │             ICD code written in the text, laterality / negation / history cues
   ▼
candidate retrieval (per system, independent)
   ├─ dictionary: TF-IDF char 3-4-grams (typos, OCR) + word 1-2-grams (exact terms) over all descriptions
   ├─ dictionary dense: IDF-weighted biomedical word vectors (scispacy en_core_sci_lg) — synonyms
   ├─ training memory: char + dense index over training entities, gold displays and the other
   │                   system's gold display for the same mention (local coding convention)
   ├─ negated mentions also query "no X" / "X absent" (SNOMED has negative concepts)
   └─ ICD hierarchy: "unspecified" leaves of retrieved categories, initial-encounter (A) injury codes
   ▼
LambdaRank reranker (LightGBM, trained out-of-fold, grouped by chunk)
   features: lexical/dense scores and ranks (absolute and relative to the best candidate), memory
             votes, gated code priors, leaf/unspecified + per-category "unspecified" convention, token and
             soft (vector) coverage, head-noun match, laterality, context overlap/similarity,
             negation/history, SNOMED semantic tag
   ▼
optional LLM adjudication of low-confidence terms (chooses among the top-8 candidates only)
   ▼
icd10cm_code, snomed_code (+ confidence, margin, top-5 alternatives)
```

ICD and SNOMED are predicted by separate retrievers and rankers. Neither prediction is derived from the other through a crosswalk.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh        # deps, scispacy vectors (~530 MB download), dictionary caches + indexes (~6 min)
cp .env.example .env         # optional keys; nothing is required for the core engine
```

Everything built is cached in `data/cache/` (gitignored).

## Run

| What | Command |
|---|---|
| Predict (single command) | `python -m scripts.predict --input eval.csv --output predictions.csv [--details details.csv] --selector data/cache/stages/v2/selector_f0` |
| Retrain from scratch | add `--refit` (~20 min on one A6000, mostly cross-encoder training; run it once without `--selector` so CE training and the 7B model never share the GPU) |
| Hold-out evaluation | `python -m scripts.evaluate` → prints metrics, writes `reports/val_predictions.csv` |
| API | `uvicorn api.main:app --port 8000`, then open `/docs` |

The input file can use the training schema (`entity, chunk_text`) or the problem schema (`chunk_id, clinical_entity[, chunk_text]`). The output columns are `chunk_id, clinical_entity, icd10cm_code, snomed_code`.

The engine is deterministic: there's no sampling, the seeds are fixed, and folds come from a hash of the chunk text. Two runs of the shipped engine on 500 terms give byte-identical output. Speed (A6000): 500 terms in 214.5 s with the selector (≈ 0.43 s/term), 32.7 s without it.

### API endpoints
- `POST /map` with `{"entity": "ms", "chunk_text": "…mitral valve area 1.2 cm2…"}` returns both codes, their confidence, and the top-5 candidates.
- `POST /map/batch`: the same for a list of terms.
- `GET /search?q=htn&system=snomed`: raw dictionary retrieval.
- `GET /codes/{icd10cm|snomed}/{code}`: dictionary entry, billable leaf or semantic tag, and training frequency.
- `GET /terminology/concept/{icd10cm|snomed}/{code}`, `GET /terminology/crosswalk/{system}/{code}`, `GET /terminology/search?q=`, `GET /terminology/versions`: our own terminology service over open sources. It replaces UMLS lookups; see [`docs/architecture.md`](docs/architecture.md).
- `GET /external/clinicaltables/{icd10cm|conditions}?q=` and `GET /external/umls/{search|crosswalk}`: proxies to public sources. These send the query to a third party, so never use them with chart text.

## Current results (5-fold CV grouped by chunk, all 5,005 DiagnosisCode rows; full log in [docs/experiments.md](docs/experiments.md))

**Shipped engine (v2):** hybrid retrieval (char/word TF-IDF, scispacy vectors, SapBERT, training memory, open-synonym aliases, ICD hierarchy) → LambdaRank → cross-encoder (top 30, trained on out-of-fold ranker lists) → local LLM selector (Qwen2.5-7B-Instruct + LoRA, trained on folds 1–4) → stacker + exact training-term lookup. The LLM runs locally, so no data leaves the machine.

| version | ICD exact | SNOMED exact | headline (min) |
|---|---|---|---|
| lexical + learned ranker (20% split) | 0.525 | 0.517 | 0.517 |
| + SapBERT retrieval, LambdaRank, open aliases, cross-system memory | 0.647 | 0.580 | 0.580 |
| + cross-encoder reranker (locked baseline) | 0.658 | 0.604 | 0.604 |
| **v2 (no selector), 5-fold** | **0.665** | **0.622** | **0.622** |
| v2 + LLM selector (shipped), **fold 0 only** | 0.716 | 0.685 | 0.685 |

- v2 vs baseline over all 5,005 rows (paired): ICD +1.1 pt (p=0.10), SNOMED +1.6 pt (p=0.01). On fold 0 the selector adds +4.4 / +8.0 pt over v2 and is +6.0 / +7.8 pt over the baseline (both p<0.001). It is measured on one fold only, because training it costs ~72 min per fold.
- Codes unseen in training (v2, 5-fold): ICD 0.431, SNOMED 0.495. ICD right 3-character category: 0.774.
- The gold code is among the candidates for 92% (ICD) / 89% (SNOMED) of terms.
- Answering only the most confident 80% of terms (v2, 5-fold): ICD 0.770 / SNOMED 0.719.
- Reproduce: `python -m scripts.pipeline --run v2 --folds all --stages s2,s3,s5` (GPU; stages are cached), selector: `python -m scripts.train_selector --run v2 --fold 0`, then `--stages s5 --with-selector --export-stacker`.
- Older locked-pipeline reproduction: `python -m scripts.evaluate --cv 5` (the cross-encoder turns on automatically with a GPU; `--ce off` to skip it).

Tried and not shipped (numbers in the experiment log): prompted local LLM adjudication (Qwen2.5-14B/32B, only +2–3 as a gated tie-breaker on one split; replaced by the fine-tuned selector), SapBERT fine-tuning, synonym-augmented cross-encoder, joint ICD+SNOMED decoding, polarity/drift features. See [docs/writeup.md](docs/writeup.md) for the write-up.

Fast ranker experiments: `python -m scripts.ranker_lab dump` once per retrieval change, then `python -m scripts.ranker_lab run`.

## LLM adjudication (optional, off by default)
Set in `.env` or as environment variables: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` (optionally `GEMINI_MODEL`). For the GPU box, use `LLM_PROVIDER=openai_compat` with `OPENAI_COMPAT_BASE_URL` and `OPENAI_COMPAT_MODEL` (vLLM, Ollama, llama.cpp).

Only terms whose ranker confidence is below `--llm-threshold` (default 0.8) are sent, with the top 15 candidates; the pick is applied only if the ranker gives it at least 0.3 × the top probability (`TERMMAP_LLM_GATE`). Temperature is 0, and every response is cached in `data/cache/llm/` so reruns are identical.

- Measure: `python -m scripts.evaluate --llm`. Early Gemini 2.5 Flash test on 200 held-out terms: ICD 0.520 → 0.610, SNOMED 0.545 → 0.655 (hosted API, so not used further).
- **Data caveat:** hosted free tiers may retain prompts, which the dataset terms forbid. Use them only for small experiments, and run the final pipeline on a self-hosted model.

## Public resources

| Resource | Use | Status |
|---|---|---|
| CDC/NCHS ICD-10-CM FY2026 tabular + index XML (public domain) | inclusion terms and index paths as extra ICD aliases | `python -m scripts.ingest cdc --year 2026` |
| Disease Ontology (CC0), Mondo (CC BY 4.0), HPO (attribution) | synonyms, open SNOMED↔ICD crosswalk, `/terminology/*` | `python -m scripts.ingest obo` |
| NLM SNOMED CT → ICD-10-CM map (UMLS licence) | agreement prior between the two predictions | put the TSV in `data/external/`, then `python -m scripts.ingest map` |
| UMLS UTS REST API (UMLS licence) | synonyms, crosswalk exploration | `UMLS_API_KEY` in `.env` |
| NLM Clinical Table Search (keyless) | consumer-name aliases for ICD | `termmap/sources/clinical_tables.py` |
| SapBERT / BioLORD-2023 / MedCPT (Hugging Face) | dense retrieval and reranking | next iteration |

All of this is ingested offline into `data/external/`. Inference never calls the network, because the dataset must not be sent to third-party services.

### Getting a UMLS key
1. Sign up at https://uts.nlm.nih.gov/uts/signup-login. You'll need an identity provider such as Login.gov, Google or Microsoft.
2. Complete the UMLS Metathesaurus License request (research use). Approval usually takes 1–3 business days.
3. Go to UTS → *My Profile* → copy the **API Key** into `.env` as `UMLS_API_KEY`.
4. Download the *SNOMED CT to ICD-10-CM Map* human-readable TSV from the NLM licensed-content downloads and put it in `data/external/`.

### Cloud dev container network
To use the public sources from a Claude Code cloud session, allowlist these hosts in the environment's network settings: `huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`, `us.aws.cdn.hf.co` (model weights), `raw.githubusercontent.com` (ontologies), `uts-ws.nlm.nih.gov`, `clinicaltables.nlm.nih.gov`, `ftp.cdc.gov`, and `generativelanguage.googleapis.com` (if testing Gemini).

## Layout
```
termmap/   config, data loading, normalize, retrieve, engine (features + ranker), sources/ (public APIs)
scripts/   predict.py (submission), evaluate.py (hold-out metrics), ingest.py (public data)
api/       FastAPI service
docs/      analysis.md (data findings, answers to open questions)
```
