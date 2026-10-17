# TermMap architecture: what exists and how it fits together

## 1. What the engine does
- **Input:** one row per extracted term: `entity` (verbatim, noisy) plus `chunk_text` (the surrounding chart text).
- **Output:** one ICD-10-CM code and one SNOMED CT concept id. Both always come from the supplied dictionaries.

Each system is predicted by its own retrievers and its own ranker. Neither prediction is derived from the other.

```
entity + chunk_text
  │
  ├─ 1. normalize (termmap/normalize.py)
  │     lowercase, unify separators, strip stray punctuation; abbreviation expansion (ambiguous → several
  │     readings, e.g. ms → multiple sclerosis | mitral stenosis); ICD code written in the text; laterality,
  │     negation and history cues from the words just before the entity in its chunk
  │
  ├─ 2. candidate retrieval, per system (termmap/retrieve.py, termmap/embed.py, termmap/engine.py)
  │     a. dictionary char 3–4-gram TF-IDF   → robust to typos / OCR ("chronickidney disease stag 3")
  │     b. dictionary word 1–2-gram TF-IDF   → exact clinical words
  │     c. dictionary word-vector dense      → synonyms (nose≈nasal, discomfort≈pain); scispacy vectors
  │     d. dictionary SapBERT dense          → concept-level similarity (optional, needs torch)
  │     e. training memory (char + dense + SapBERT) over training entities, gold displays and the other
  │        system's gold display for the same mention → how HiLabs coders actually code
  │     f. ICD hierarchy expansion → "unspecified" leaves of retrieved categories; 7th-char "A" for injuries
  │     g. negated mentions also search "no X" / "X absent" (SNOMED has negative concepts)
  │     The dictionaries are enriched with open-source aliases (section 3) before indexing.
  │
  ├─ 3. features per (term, candidate) (termmap/engine.py::_featurize, termmap/rank.py::derive)
  │     ~70 features: scores and ranks from every source, gaps to the best candidate of the same term,
  │     agreement count, token and soft (vector) coverage, head-noun match, laterality agreement,
  │     context overlap/similarity, negation/history × negative/history concepts, training frequency
  │     prior (gated by textual evidence), leaf / unspecified flags, per-category "unspecified" rate,
  │     SNOMED semantic tag, open-alias hit
  │
  ├─ 4. LambdaRank ranker (LightGBM, termmap/rank.py) → probability per candidate (softmax per term)
  │     trained out-of-fold: the training memory used for a row never contains that row's chunk
  │
  └─ 5. optional LLM adjudication (termmap/llm.py) for terms whose top probability < 0.6:
        the LLM sees the term, a context window and the top-8 candidates, and must pick one of them.
        Temperature 0, responses cached on disk. Off unless LLM_PROVIDER is set.
```

## 2. Data we were given and how it's used

| File | What it is | How we use it |
|---|---|---|
| `hackathon_train.csv` | 9,646 labelled rows; 5,005 are DiagnosisCode (2,305 ICD + 2,700 SNOMED) | ranker training labels; training memory (aliases + priors); evaluation |
| `icd10cm_codes.xlsx` | 98k ICD-10-CM codes (FY2026), multiple descriptions each | legal answer space + retrieval index |
| `SNOMEDCT_US_codes_and_descriptions.xlsx` | 382k SNOMED concepts, 985k descriptions (HiLabs' gte-base index export) | legal answer space + retrieval index; drug/product/etc. tags excluded from the index |

## 3. Our own terminology service (instead of the UMLS API)

`python -m scripts.ingest all` downloads the freely licensed sources once and builds `data/external/terminology/*.parquet`. Inference never calls the network.

| Source | Licence | What we take |
|---|---|---|
| CDC/NCHS ICD-10-CM FY2026 tabular + index | public domain | inclusion terms, index paths ("Hypertension, hypertensive" → I10) as ICD aliases; Excludes notes stored |
| Disease Ontology (DOID, 2026-09-30) | CC0 | labels, synonyms, ICD10CM + SNOMEDCT_US + UMLS cross-references |
| Mondo | CC BY 4.0 | labels, synonyms, ICD10CM + SCTID (SNOMED) + UMLS + DOID cross-references |
| HPO (2026-09-01) | HPO licence (attribution) | sign/symptom names and synonyms, linked by exact name match |

How the sources are linked (`termmap/terminology.py`):
1. Mondo and DOID concepts are merged where Mondo asserts equivalence, or where they share a UMLS CUI.
2. A merged concept links to a dictionary code through an ontology cross-reference, or through an exact name match against the dictionary (only when that name maps to at most 2 codes).
3. Aliases are emitted only for concepts linked to at most 3 codes, so broad groupings don't pollute retrieval.

What the store contains:
- 64k concepts and 204k synonyms.
- 145k aliases: 54k for ICD and 92k for SNOMED.
- An open SNOMED↔ICD crosswalk of 9,856 pairs.
- Coverage: 33% of training ICD gold codes and 42% of SNOMED gold codes gain extra synonyms.

The endpoints (`api/main.py`) replace the UMLS lookups we'd otherwise need:
- `GET /terminology/concept/{icd10cm|snomed}/{code}`: synonyms from every source, linked concepts, related codes in the other system.
- `GET /terminology/crosswalk/{system}/{code}`: open crosswalk with provenance. It's a feature, not ground truth.
- `GET /terminology/search?q=`: synonym search returning linked ICD and SNOMED codes.
- `GET /terminology/versions`: the source releases used.

## 4. What the numbers mean

We hold out 20% of the labelled chunks (`scripts/evaluate.py::split`, grouped by `chunk_text` so no context leaks) and train on the other 80%. The terms that land in the 20% are new to the engine, which is how it would see the real evaluation file.

- **exact**: the fraction of held-out terms whose top-1 predicted code equals the gold code. This is the challenge's primary metric. The headline is min(ICD, SNOMED).
- **candidate recall**: the fraction where the gold code is anywhere in the candidate list. It's the ceiling for the ranker.
- **seen / unseen**: exact accuracy split by whether the gold code occurred in the training part. Unseen is the long-tail criterion (20% of the judging).
- **category** (ICD only): the predicted code is in the right 3-character category (partial credit).
- **accuracy at X% coverage**: accuracy if we abstain on the least confident (100−X)% of terms.

The sample is 469 ICD terms and 565 SNOMED terms, so ±2 points is noise. Final numbers should use 5-fold cross-validation over all rows.

## 5. Commands
| Command | Purpose |
|---|---|
| `bash scripts/setup.sh [--ingest] [--with-sapbert]` | install, download vectors/sources, build indexes |
| `python -m scripts.ingest all` | refresh the open sources |
| `python -m scripts.evaluate [--llm]` | hold-out metrics + `reports/val_predictions.csv` |
| `python -m scripts.ranker_lab dump` / `run` / `llm 0.6 200` | fast ranker / LLM experiments on cached features |
| `python -m scripts.predict --input X.csv --output Y.csv [--details D.csv]` | submission run |
| `uvicorn api.main:app` | REST API (`/docs` for the interactive UI) |

## 6. Code map
| Path | Role |
|---|---|
| `termmap/config.py` | paths, constants, `.env` loading |
| `termmap/data.py` | load CSV and xlsx (cached as parquet), ICD leaf derivation, SNOMED semantic tags |
| `termmap/normalize.py` | cleaning, abbreviations, negation/history/laterality cues |
| `termmap/retrieve.py` | TF-IDF indexes and the per-system dictionary retriever |
| `termmap/embed.py` | scispacy word-vector index, SapBERT encoder/index |
| `termmap/engine.py` | code systems, training memory, candidate gathering, features, fit/rank/predict |
| `termmap/rank.py` | relative features, LightGBM LambdaRank, softmax confidence |
| `termmap/llm.py` | optional Gemini / OpenAI-compatible adjudicator with disk cache |
| `termmap/terminology.py` | open terminology store and lookups |
| `termmap/sources/` | clients: CDC, OBO ontologies, NLM Clinical Tables, UMLS (optional), NLM map loader |
| `api/main.py` | FastAPI service |
| `scripts/` | setup, ingest, evaluate, ranker_lab, predict |
