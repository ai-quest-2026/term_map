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
| Predict (single command) | `python -m scripts.predict --input eval.csv --output predictions.csv [--details details.csv]` |
| Retrain from scratch | add `--refit` (about 2 min on 4 CPUs) |
| Hold-out evaluation | `python -m scripts.evaluate` → prints metrics, writes `reports/val_predictions.csv` |
| API | `uvicorn api.main:app --port 8000`, then open `/docs` |

The input file can use the training schema (`entity, chunk_text`) or the problem schema (`chunk_id, clinical_entity[, chunk_text]`). The output columns are `chunk_id, clinical_entity, icd10cm_code, snomed_code`.

The engine is deterministic: there's no sampling, the seeds are fixed, and folds come from a hash of the chunk text.

### API endpoints
- `POST /map` with `{"entity": "ms", "chunk_text": "…mitral valve area 1.2 cm2…"}` returns both codes, their confidence, and the top-5 candidates.
- `POST /map/batch`: the same for a list of terms.
- `GET /search?q=htn&system=snomed`: raw dictionary retrieval.
- `GET /codes/{icd10cm|snomed}/{code}`: dictionary entry, billable leaf or semantic tag, and training frequency.
- `GET /external/clinicaltables/{icd10cm|conditions}?q=` and `GET /external/umls/{search|crosswalk}`: proxies to public sources. These send the query to a third party, so never use them with chart text.

## Current results (grouped 80/20 hold-out on DiagnosisCode rows, no LLM)

| version | ICD exact | SNOMED exact | ICD cand. recall | SNOMED cand. recall | ICD unseen | SNOMED unseen |
|---|---|---|---|---|---|---|
| v0.1 lexical + HGB | 0.525 | 0.517 | 0.849 | 0.823 | 0.238 | 0.348 |
| + word-vector dense | 0.529 | 0.515 | 0.870 | 0.857 | 0.248 | 0.362 |
| + LambdaRank, relative features | 0.533 | 0.535 | 0.866 | 0.857 | 0.257 | 0.375 |
| + paired-display memory (lab) | 0.561 | 0.535 | 0.870 | 0.857 | 0.317 | 0.394 |

With 469 ICD and 565 SNOMED validation terms, differences under about 2 points are noise.

When the gold code is among the candidates, the ranker picks it only about 62% of the time. Many of the misses are close calls that need clinical judgement ("neck discomfort" → neck pain vs discomfort). The next levers are LLM adjudication of low-confidence terms, and a sentence-level biomedical encoder (SapBERT / BioLORD) once Hugging Face is reachable.

Fast ranker experiments: `python -m scripts.ranker_lab dump` once per retrieval change, then `python -m scripts.ranker_lab run`.

## LLM adjudication (optional, off by default)
Set in `.env` or as environment variables: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` (optionally `GEMINI_MODEL`). For the GPU box, use `LLM_PROVIDER=openai_compat` with `OPENAI_COMPAT_BASE_URL` and `OPENAI_COMPAT_MODEL` (vLLM, Ollama, llama.cpp).

Only terms whose top candidate has a probability below 0.6 are sent. The model must pick one of the top 8 candidates. Temperature is 0, and every response is cached in `data/cache/llm/` so reruns are identical.

- Measure: `python -m scripts.ranker_lab llm 0.6 200`, or `python -m scripts.evaluate --llm`.
- **Data caveat:** hosted free tiers may retain prompts, which the dataset terms forbid. Use them only for small experiments, and run the final pipeline on a self-hosted model.

## Public resources

| Resource | Use | Status |
|---|---|---|
| CDC/NCHS ICD-10-CM FY2026 tabular + index XML (public domain) | inclusion terms and index paths as extra ICD aliases | `python -m scripts.ingest cdc --year 2026` |
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
To use the public sources from a Claude Code cloud session, allowlist these hosts in the environment's network settings: `huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`, `uts-ws.nlm.nih.gov`, `clinicaltables.nlm.nih.gov`, `ftp.cdc.gov`, and `generativelanguage.googleapis.com` (if testing Gemini).

## Layout
```
termmap/   config, data loading, normalize, retrieve, engine (features + ranker), sources/ (public APIs)
scripts/   predict.py (submission), evaluate.py (hold-out metrics), ingest.py (public data)
api/       FastAPI service
docs/      analysis.md (data findings, answers to open questions)
```
