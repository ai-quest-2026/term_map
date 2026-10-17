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
  4. cross-encoder reranker (SapBERT/PubMedBERT weights) reads [term ; context sentence] × [candidate
     description (tag) [ICD category]] for the top 15, trained listwise on our hard negatives;
     final score = 0.7 · z(ranker) + 0.3 · z(cross-encoder)
  ▼
icd10cm_code, snomed_code + calibrated confidence, margin and top-5 alternatives
```

There is one deterministic command: `scripts/predict.py`. Seeds are fixed, LightGBM is deterministic, and nothing calls the network at inference. Speed is about 35 ms per term on one GPU.

## 3. Results (5-fold CV, all labelled diagnosis terms)

| Stage | ICD | SNOMED | Headline |
|---|---|---|---|
| Lexical retrieval + learned ranker | 0.525* | 0.517* | 0.517 |
| + SapBERT retrieval, LambdaRank, open-synonym aliases, memory of the other system's wording | 0.647 | 0.580 | 0.580 |
| **+ cross-encoder reranker (final)** | **0.658** | **0.604** | **0.604** |

*20% split.

- **Codes unseen in training (long tail):** ICD 0.440, SNOMED 0.477.
- **ICD right 3-character category:** 0.770.
- **The right code is among the candidates** for 93% (ICD) and 92% (SNOMED) of terms.
- **Confidence is usable for abstention.** Answering only the most confident 80% of terms gives ICD 0.753 / SNOMED 0.687. Terms scored ≥ 0.9 are 94–95% correct.

## 4. Design decisions
- **Retrieve-then-rank, not classification.** Half the SNOMED test codes are unseen, so a classifier over training codes would cap long-tail accuracy near zero.
- **Hybrid retrieval.** Each source fixes a different failure:
  - character n-grams handle OCR ("chronickidney disease stag 3");
  - word vectors handle lay synonyms (nose ≈ nasal);
  - SapBERT handles concept-level synonymy.

  SapBERT was the largest single gain: +8 ICD / +4 SNOMED.
- **Out-of-fold training.** The ranker is trained on features computed with a training-example memory that excludes the term's own fold. Otherwise memory features look perfect in training and fail at test time.
- **Leakage discipline for neural parts.** The cross-encoder is retrained inside every fold. The SapBERT variant fine-tuned on labelled terms was *not* shipped, because it would leak into the ranker's training features.
- **No LLM in the final system.** The data may not go to retaining APIs, so only local models were acceptable. A local Qwen2.5-32B was less accurate than our ranker on the terms it was asked about (0.41 vs 0.45). As a gated tie-breaker it added ≈ +2–3 points on one split, but that is not confirmed in 5-fold, and it would cost about a second per term. The code path is kept (`LLM_PROVIDER`) and documented.

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

**Label consistency sets the ceiling.** We grouped terms whose words match after dropping qualifiers: 28% of such ICD groups and 43% of SNOMED groups carry more than one gold code. Part of this is legitimate (laterality, acuity), but some is the same meaning coded two ways:
- "class iii obesity" → *Severe obesity*, but "obesity class iii" → *Obese class III*;
- "asthma with rsv" → J45.901, but "rsv with asthma" → J45.909;
- "ble edema" → R60.9, but "mild ble edema" → R60.0.

No mapper can be right on both sides of such pairs. We think adjudicated, multi-annotator labels (the PS's open question) are a prerequisite for a 95% target.

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
  - Qwen2.5-14B/32B locally: +2–3 only as a gated tie-breaker.

## 8. What we would do next
1. Get the label conventions adjudicated, or obtain multi-annotator labels, to know the true ceiling.
2. Train the cross-encoder on more in-domain labelled data, ideally with the full chart section rather than a 120-character window.
3. Fine-tune a local LLM on the training conventions as a reranker, instead of zero-/few-shot prompting.
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
| Qwen2.5-14B/32B-Instruct-AWQ via vLLM (experiments only, local) | LLM adjudication tests | Apache-2.0 |
| Gemini 2.5 Flash (preliminary test on 200 held-out terms only) | LLM adjudication test | Google API terms |
