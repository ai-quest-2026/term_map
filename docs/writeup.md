# TermMap: mapping clinical terms to ICD-10-CM and SNOMED CT

*Team submission write-up. Every number is reproducible with the commands in the README; the full experiment log is in `docs/experiments.md`.*

## 1. Problem and how we framed it
Each input is an extracted clinical entity plus the chunk of chart text around it. The output is one ICD-10-CM code and one SNOMED CT concept, both from the supplied dictionaries (98k ICD codes; 382k SNOMED concepts, 985k descriptions).

We treat this as **two independent entity-linking problems** (one per system), each solved as *retrieve candidates → rank them*. Three facts from the data shaped the design:

- **Every entity string occurs once per system** in the training file, so memorising strings is worth nothing. The engine has to generalise.
- **24% of ICD and 48% of SNOMED held-out gold codes never occur in training.** Retrieval over the full dictionaries, not classification over seen codes, is mandatory. This is also the long-tail criterion.
- **The labels encode local convention.** For example, the same term gets ICD "unspecified" vs. specific codes, or a SNOMED "No …" concept when the chart negates the finding. The training file teaches *how* coders choose, which is why it drives the ranker and the training-example memory rather than a code classifier.

**Evaluation.** 5-fold cross-validation grouped by chunk, so no surrounding text leaks between train and test, over all 5,005 labelled diagnosis rows. Metric: exact match per system; headline = the lower of the two. Ablations used a fixed 20% grouped split, and every shipped change was confirmed by 5-fold.

## 2. Architecture

```
entity + chunk_text
  │ 1. normalise: lowercase, punctuation/OCR clean-up, abbreviation expansion (ambiguous ones → several
  │    readings), ICD code written in the text, laterality / negation / history cues
  ▼
  2. candidate retrieval, per system (union, ~80 candidates)
     ├─ character 3–4-gram TF-IDF (typos, OCR)         ├─ word 1–2-gram TF-IDF (exact clinical words)
     ├─ biomedical word vectors (scispacy)             ├─ SapBERT dense retrieval (concept similarity)
     ├─ training-example memory: char / vectors / SapBERT over labelled terms + gold displays
     │  (+ the other system's gold display for the same mention)
     ├─ open-synonym aliases: CDC ICD-10-CM index, Disease Ontology, Mondo, HPO (our own terminology service)
     └─ ICD hierarchy: "unspecified" leaves, initial-encounter (A) variants, parents of newly split codes
  ▼
  3. LambdaRank (LightGBM) over ~90 features: every source's score / rank / gap to the best candidate,
     coverage and soft (vector) coverage, laterality, context overlap, negation × negative concepts,
     training-frequency prior gated by textual evidence, leaf / unspecified flags, SNOMED semantic tag
     → trained out-of-fold, so memory features look on training terms exactly as on unseen terms
  ▼
  4. cross-encoder reranker (SapBERT weights) reads [term ; context sentence] × [candidate
     description (tag) [ICD category]] for the top 30, trained listwise on out-of-fold ranker lists
  ▼
  5. LLM selector: local Qwen2.5-7B-Instruct + LoRA adapter, fine-tuned to pick the gold code among
     the ranked candidates (letter choice); its per-candidate probability is a feature
  ▼
  6. stacker (LambdaRank over ranker / cross-encoder / selector scores, ranks and gaps)
     + exact training-term lookup
  ▼
icd10cm_code, snomed_code + calibrated confidence, margin and top-5 alternatives
```

There is one deterministic command: `scripts/predict.py`. Seeds are fixed, LightGBM is deterministic, cross-encoder training uses deterministic GPU kernels, and nothing calls the network at inference. Retrieval and ranking take about 25–30 ms per term per system on CPU. End-to-end, the shipped engine maps 500 terms in 214.5 s on one RTX A6000 (≈ 0.43 s per term for both codes, including loading the 7B model); without the selector the same 500 take 32.7 s (≈ 65 ms per term). Two runs give byte-identical output. The selector is a local open-weight model: no data leaves the machine.

## 3. Results (5-fold CV, all labelled diagnosis terms)

| Stage | ICD | SNOMED | Headline |
|---|---|---|---|
| Lexical retrieval + learned ranker | 0.525* | 0.517* | 0.517 |
| + SapBERT retrieval, LambdaRank, open-synonym aliases, memory of the other system's wording | 0.647 | 0.580 | 0.580 |
| + cross-encoder reranker (locked baseline) | 0.658 | 0.604 | 0.604 |
| **v2: fixed displays/tags, OOF-trained cross-encoder (top 30), stacker, term lookup** | **0.665** | **0.622** | **0.622** |
| v2 + LoRA LLM selector (shipped): **fold 0 only** | 0.716† | 0.685† | 0.685† |

*20% split. †One fold (451 ICD / 540 SNOMED terms); on the same fold v2 without the selector scores 0.667 / 0.617 and the baseline 0.656 / 0.607. The selector is trained per fold (~72 min on one A6000), so the 5-fold run measures v2 *without* it.

- **End-to-end check of the submission command (blind fold 0):** answers hidden, engine refit on folds 1–4, `scripts/predict.py` run as the judges would: schema OK, 576 terms in 231.5 s, ICD 0.796 / SNOMED 0.772. This is optimistic, because the shipped stacker was fitted on fold 0's labels; the clean fold-0 figures are 0.716 / 0.685. With the entity alone (no chunk text) it gives 0.716 / 0.663.
- **v2 vs the baseline (paired over all 5,005 rows):** ICD +1.1 pt [−0.2, +2.4], p=0.10; SNOMED +1.6 pt [+0.4, +2.8], p=0.01.
- **The selector vs the baseline on fold 0:** ICD +6.0 pt [+2.7, +9.3], SNOMED +7.8 pt [+4.3, +11.3], both p<0.001. The largest gain is on SNOMED codes unseen in training (+12.3 pt).
- **Codes unseen in training (long tail), v2 5-fold:** ICD 0.431, SNOMED 0.495 (fold 0 with the selector: 0.474 / 0.605).
- **ICD right 3-character category:** 0.774 (fold 0 with the selector: 0.834).
- **The right code is among the candidates** for 92% (ICD) and 89% (SNOMED) of terms.
- **Confidence is usable for abstention.** v2 5-fold, answering only the most confident 80% of terms: ICD 0.770 / SNOMED 0.719; terms scored ≥ 0.9 (28–34% of terms) are 93–94% correct. Fold 0 with the selector: 0.822 / 0.775 at 80%, and 95–96% correct at ≥ 0.9.

## 4. Design decisions
- **Retrieve-then-rank, not classification.** Half the SNOMED test codes are unseen, so a classifier over training codes would cap long-tail accuracy near zero.
- **Hybrid retrieval.** Each source fixes a different failure:
  - character n-grams handle OCR ("chronickidney disease stag 3");
  - word vectors handle lay synonyms (nose ≈ nasal);
  - SapBERT handles concept-level synonymy.

  SapBERT was the largest single gain: +8 ICD / +4 SNOMED.
- **Out-of-fold training.** The ranker is trained on features computed with a training-example memory that excludes the term's own fold. Otherwise memory features look perfect in training and fail at test time.
- **Leakage discipline for neural parts.** The cross-encoder is retrained inside every fold. The SapBERT variant fine-tuned on labelled terms was *not* shipped, because it would leak into the ranker's training features.
- **A fine-tuned local LLM as a selector, not a zero-shot judge.** The data may not go to retaining APIs, so only local models were acceptable. Prompted LLMs (Qwen2.5-14B/32B) were weaker than our ranker and only helped as a gated tie-breaker (+2–3 on one split). Fine-tuning Qwen2.5-7B-Instruct with LoRA (0.5% of weights, ~70 min per fold) to choose among the ranker's candidates changed that: on fold 0 the selector alone beats the ranker (0.667 vs 0.641 ICD, 0.663 vs 0.578 SNOMED), and as a stacker feature it adds +4.4 / +8.0 pt over v2. It is an open-weight model running on our GPU, so no data leaves the machine. The shipped adapter is the fold-0 one (trained on folds 1–4); its gain is measured on one fold only.

## 5. Error analysis
Using the 5-fold predictions (`scripts/error_report.py`):

**Most errors are choices, not misses.** About 81% of ICD and SNOMED errors had the gold code in the candidate list but ranked lower. Only ~7–8% of terms never had the gold code retrieved.

| Failure mode | Evidence | Example (term → gold / predicted) |
|---|---|---|
| Specificity | 31% of ICD errors are in the right 3-char category. Wrong-direction errors split evenly: 63 too general, 54 too specific. | constipation → K59.00 / K59.09; obesity class ii → E66.9 / E66.812 |
| Context decides organ or sense | | acute infarct → I63.9 (brain) / I21.9 (heart); "ms", "lad" |
| SNOMED negative concepts from context | | "acute changes" → *No status change*; "focal neurologic symptoms" → *No focal neurology* |
| Noisy / OCR terms | SNOMED accuracy 0.42 on terms with non-dictionary tokens (10% of terms) | "mucusua" → *Slime in urine* |
| Version drift | gold codes that FY2024–26 split into children: 26% accuracy (1.9% of ICD rows) | M51.36 (gold) vs FY2026 M51.360… |
| Long tail | unseen codes 0.44 / 0.48 vs seen 0.73 / 0.72 | |

**Labels are mostly consistent, so the gap is ours to close.** Training terms with exactly the same words agree on the code 97% of the time for ICD and 91% for SNOMED. Disagreements are rare but real:
- "class iii obesity" → *Severe obesity*, but "obesity class iii" → *Obese class III*;
- "asthma with rsv" → J45.901, but "rsv with asthma" → J45.909.

(An earlier draft reported 28% / 43% conflicting groups. That grouping dropped qualifiers such as "mild" or "acute", which legitimately change the code.) The real limits are retrieval of codes never seen in training, and decisions that need the context, which our first architecture barely used: blanking the chart text changed accuracy by under 1 point.

## 6. Innovations
- **Our own terminology service instead of the UMLS API.** It merges Disease Ontology (CC0), Mondo (CC BY 4.0) and HPO with the CDC ICD-10-CM FY2026 index. The result is 64k concepts, 145k extra aliases and a 9,856-pair open SNOMED↔ICD crosswalk, served by `/terminology/{concept,crosswalk,search}`. It covers 33% of ICD and 42% of SNOMED gold codes with extra synonyms, with no licence wait.
- **Cross-system memory.** When a mention is coded in both systems, the other system's gold wording becomes a training alias. This gave +2.8 ICD on the split.
- **Routing diagnostics.** The LLM is evaluated per confidence bucket, comparing several routing rules. This showed unrestricted LLM overrides *hurt* (−4 to −6), and a gated tie-breaker is the only safe use.

## 7. Tried and abandoned (all logged with numbers)
- **Hand-made features:**
  - negation/polarity conflict, version-drift parents, more abbreviations: neutral in 5-fold;
  - joint ICD+SNOMED decoding from pair co-occurrence: +1.1 ICD, −0.7 SNOMED.
- **SapBERT fine-tuning:**
  - on labelled terms: +13 ICD / +6 SNOMED recall@15, but almost all of it on codes already seen in training, which the memory covers, and it leaks into the ranker;
  - on open synonyms only: neutral.
- **Cross-encoder variants:** distant supervision from 10k synonyms per system made it stronger alone (+4.7 / +2.0) but not in the blend.
- **LLM adjudication:**
  - Gemini 2.5 Flash: +9 / +11 on 200 terms, but it is a hosted API that may retain data, so it was used only for that preliminary test;
  - Qwen2.5-14B/32B locally, prompted: +2–3 only as a gated tie-breaker; replaced by the fine-tuned 7B selector (section 4).

## 8. What we would do next
1. Get the label conventions adjudicated, or obtain multi-annotator labels, to know the true ceiling.
2. Train the cross-encoder on more in-domain labelled data, ideally with the full chart section rather than a 120-character window.
3. Train the LLM selector on every fold (and on all data for the shipped adapter) to confirm its fold-0 gain in 5-fold CV; it costs ~72 min per fold on one A6000.
4. Add the official SNOMED→ICD-10-CM map once the UMLS licence arrives, as an agreement feature.

## 9. External resources and licences

| Resource | Use | Licence |
|---|---|---|
| SapBERT-from-PubMedBERT-fulltext (cambridgeltl) | dense retrieval, cross-encoder initialisation | Apache-2.0 |
| scispacy en_core_sci_lg 0.5.4 vectors | word-vector retrieval | CC BY-SA 3.0 |
| CDC/NCHS ICD-10-CM FY2026 tabular + index | aliases, inclusion terms | public domain |
| Disease Ontology (2026-09-30) | synonyms, cross-references | CC0 1.0 |
| Mondo (downloaded 2026-10-17) | synonyms, cross-references | CC BY 4.0 |
| Human Phenotype Ontology (2026-09-01) | sign/symptom synonyms | HPO licence (attribution) |
| LightGBM, scikit-learn, PyTorch, transformers, FastAPI | modelling, serving | MIT / BSD / BSD / Apache-2.0 / MIT |
| Qwen2.5-7B-Instruct + our LoRA adapter (local, shipped) | LLM selector | Apache-2.0 |
| Qwen2.5-14B/32B-Instruct-AWQ via vLLM (experiments only, local) | LLM adjudication tests | Apache-2.0 |
| Gemini 2.5 Flash (preliminary test on 200 held-out terms only) | LLM adjudication test | Google API terms |
