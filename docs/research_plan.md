# Research & experiment plan (ML + DL + retrieval)

## Context
The baseline is a hybrid lexical/dense retrieval + LambdaRank + optional LLM pipeline, scoring about 0.55 / 0.53 exact on the hold-out, or 0.61 / 0.655 with Gemini adjudicating low-confidence terms. The user wants a researched plan that:
- combines classical ML (classification, regression, clustering) and deep learning at the right stages with new retrieval methods and ensembles, after proper data cleaning;
- explains what the training file is for;
- grounds each sub-problem in the literature;
- defines the tests and evals that decide what ships.

**Hardware:** RTX A6000 48 GB, Xeon w7-2495X (24 cores), 255 GB RAM, WSL. Everything below fits it.

## A. What the training file is for (answer for the user)
Retrieval alone (search over dictionaries) needs no training: that's our zero-shot baseline. The 5,005 labelled diagnosis rows are used in five places:
1. **Evaluation.** It's the only gold data, so every number comes from held-out folds of it.
2. **Learning to decide.** The ranker learns which signals predict the gold code among about 80 candidates. Without labels we'd have to hand-weight them.
3. **Local coding convention.** It captures how HiLabs coders resolve ambiguity: "lad" → R59.0, "unspecified" preferences, negative SNOMED concepts. The PS says this "is learnable and is not in any public resource."
4. **Fine-tuning neural models.** Entity → gold-description pairs, with our retriever's hard negatives, adapt SapBERT and a cross-encoder to chart language.
5. **Calibration.** It sets confidence thresholds for LLM routing and abstention.

Caveat: entity strings never repeat, and about 24% of ICD and 49% of SNOMED held-out gold codes never appear in training. Training must teach *how* to choose, not *what* to memorise. Long-tail accuracy is 20% of the score.

## B. Sub-problems → evidence → technique → experiment
| # | Sub-problem (PS failure mode) | Literature signal | Technique for us | Experiment |
|---|---|---|---|---|
| 1 | Noisy surface forms (OCR, typos, truncation) | char n-gram robustness; SymSpell-style correction; LLM mention simplification raised candidate recall (+2.9 pt) on SympTEMIST (Database 2025) | keep char TF-IDF; add a SymSpell corrector built on dictionary + training vocab; optional local-LLM "rewrite to canonical clinical term" query (not decision) | recall@30 before/after on CV |
| 2 | Synonyms / nearest string is wrong (HTN vs pulmonary HTN) | SapBERT self-alignment on UMLS synonyms (Liu et al. 2021); KRISSBERT reports +10 pt over SapBERT (paper's claim); BELB shows neural methods are uneven across entity types (Garda et al. 2023) | SapBERT index (running now) → **fine-tune SapBERT** with InfoNCE on training (entity → gold description) plus open-ontology synonym pairs, with hard negatives from our candidate lists | recall@k and exact, base vs fine-tuned, grouped CV (the encoder never sees the eval fold) |
| 3 | Decision among close candidates | retrieve-then-rerank is standard; cross-encoder rerankers over SapBERT improve EL (MedCPT; OntolinkX thesis); LLM rerankers +3–24% (BeLink 2026, abstract claim) | **cross-encoder** (PubMedBERT / BiomedBERT, or MedCPT-Cross-Encoder init) on `[term ; context window] [SEP] [candidate description ; tag]`, trained with hard negatives (BCE or listwise) → its score becomes a ranker feature | exact accuracy: ranker alone vs + cross-encoder feature vs cross-encoder alone |
| 4 | Abbreviation collision (MS, RA, PID) | zero-shot LLMs and BERT disambiguate clinical acronyms well (Agrawal et al. 2022 CASI 0.86; later CASI studies) | **sense classifier:** for ambiguous abbreviations, compare the SapBERT context embedding with each expansion's embedding; feature `sense_prob`; LLM fallback | accuracy on the abbreviation slice (entities ≤5 chars or in the abbreviation table) |
| 5 | Specificity calibration (E11.9 vs E11.40, unspecified) | LLM coders make many near misses that are hierarchically close (arXiv 2510.07629); tree search over ICD helps rare codes (Boyle/Soroush et al., NeurIPS 2023 workshop) | **hierarchical classifiers:** (a) ICD 3-char category classifier (LightGBM or logistic on SapBERT(term+context) embeddings, trained OOF) → `cat_prob` feature; (b) binary "specific vs unspecified" classifier → interaction with the candidate's `unspec` flag; (c) LLM tree-descent variant for ICD on low-confidence terms | category accuracy, then exact; error slice "right category, wrong leaf" |
| 6 | Negated / historical mentions | NegEx → ConText (F1 ≈ 0.93 on UPMC notes); dependency-aware DEEPEN | replace our crude window with **medspaCy ConText** (negation, historical, hypothetical, family) → features × negative/history concepts | accuracy on negated and historical slices |
| 7 | Granularity mismatch between systems | PS: crosswalk is a prior, not truth; our open crosswalk agrees 58% with training pairs | **joint second pass:** `xwalk_support` (open crosswalk + training pair counts, fold-safe) between each candidate and the other system's top-5 → re-rank; disagreement → lower confidence → LLM routing | exact accuracy, plus calibration of "systems disagree" |
| 8 | Long tail | GenBioEL/KRISSBERT-style self-supervision; open-ontology aliases (+2.3 pt recall measured) | aliases, fine-tuned SapBERT, LLM over top-15 | unseen-code accuracy slice (20% of judging) |
| 9 | Version drift | CDC FY2026 conversion table | map old → new codes; feature "candidate is a parent of a newly split code"; report drift handling in the write-up | rows with non-leaf gold (M51.36…) |
| 10 | Calibration / abstention | PS secondary metric | **isotonic regression** on OOF top-1 probabilities → calibrated confidence; coverage–accuracy curve; route to the LLM below a tuned threshold | ECE, accuracy at 97/90/80% coverage |

## C. Data cleaning (before any model work)
1. **Label audit.**
   - Invalid codes (`L6.0` ×2) and gold codes outside the dictionary: drop from training, report.
   - Exact duplicate (entity, chunk) rows.
   - Entities that also appear as non-diagnosis rows (34).
2. **Confident-learning style noise detection.** Flag training rows where the OOF ranker puts high probability on another code *and* that code's description is an exact match to the entity. Review the top 50 manually for the write-up only; never edit eval data. Train once with and once without the flagged rows.
3. **Text normalisation.** Strip table scaffolding (`header_n:`, `::`, `<br>`), page footers and dates from `chunk_text`. Context window = the sentence containing the entity (medspaCy sentence splitter) instead of fixed ±200 chars.
4. **Entity cleaning.** Leading list markers (`•`, `-`, numbering), embedded ICD codes (extracted as a feature), trailing qualifiers ("(hcc)", "nos"), unit or number noise.
5. **Dictionary cleaning.**
   - SNOMED: drop out-of-scope tags (done), keep the FSN.
   - ICD: official billable flag from the CDC order file in place of the derived leaf flag.
   - Dedupe aliases.

## D. Retrieval stack after this plan (candidate generation; recall is the target)
- **Sparse:**
  - char TF-IDF and word TF-IDF (existing);
  - **BM25** (`bm25s`) over the same texts;
  - all of the above on SymSpell-corrected and abbreviation-expanded queries.
- **Dense:**
  - scispacy word vectors (existing);
  - SapBERT base (being built);
  - **fine-tuned SapBERT** (FAISS flat IP on GPU or CPU; ~600k vectors is trivial).
- **Memory:** kNN over training entities with every encoder (existing pattern in `Memory`, `termmap/engine.py`).
- **Structural:**
  - ICD "unspecified" siblings and 7th-char A (existing);
  - **children and parents of the top categories** when the category classifier is confident.
- **Fusion:** union for the ranker (existing). **RRF** (k = 60) as a no-learning baseline, also reported.

Targets: **recall@15 ≥ 0.92 ICD / ≥ 0.88 SNOMED** (now ~0.85 / 0.79), so the LLM's ceiling rises.

## E. Decision stack (classification / regression / ensembles)
1. **Stage classifiers** (features into the ranker, all trained out-of-fold to avoid leakage):
   - ICD category classifier;
   - SNOMED semantic-tag classifier (disorder / finding / morph / situation …);
   - specific-vs-unspecified classifier;
   - abbreviation-sense scorer.

   All are LightGBM or logistic regression over [SapBERT(term), SapBERT(context), char-TF-IDF SVD].
2. **Cross-encoder score** as a feature (GPU).
3. **LambdaRank** (existing) as the stacker over everything: the learned ensemble.
4. **Calibration:** isotonic regression on the top-1 score and margin.
5. **Joint pass:** cross-system agreement re-rank.
6. **LLM adjudicator:** local **Qwen2.5-32B-Instruct-AWQ** via vLLM (~20 GB). Llama-3.3-70B-AWQ (~40 GB) as a stretch, and Gemini only for quick comparison.
   - **When it runs:** only below the calibrated threshold.
   - **What it sees:** the top-15 candidates with descriptions, tags, the hierarchy path for ICD, and the context sentence.
   - **Robustness:** two candidate orderings plus majority vote with the ranker. Deterministic: temperature 0, seed, disk cache.
   - **ICD variant to test:** category-then-leaf tree descent.
7. **Ensemble check:**
   - final = LLM pick if it agrees with any of {ranker top-1, cross-encoder top-1}, otherwise the higher-confidence model;
   - compare with the plain LLM override.

## F. Clustering (where it actually helps)
- **Hard-negative mining:** cluster SapBERT embeddings of dictionary descriptions (FAISS k-means, ~5k clusters per system). Negatives for SapBERT and cross-encoder fine-tuning come from the gold's cluster plus our retriever's top-k.
- **Ambiguity feature:** the number of dictionary concepts within cosine 0.95 of a candidate (synonym collisions, e.g. SNOMED near-duplicates) → `crowdedness` feature, and a signal for LLM routing.
- **Error analysis:** cluster *error* rows (term + gold embedding) with HDBSCAN to find systematic failure groups for the write-up, e.g. "lab-value phrases", "cardiac echo findings".
- **Not used** for the final decision directly. Clustering isn't a mapping method here, and that's said in the write-up.

## G. Evaluation protocol (decides what ships)
- **Primary:** 5-fold CV grouped by `chunk_text` (`scripts/evaluate.py --cv 5`, fixed md5 folds); exact accuracy per system; headline = min.
- **Slices:**
  - unseen-code (long tail);
  - abbreviation;
  - negated / historical;
  - laterality-bearing;
  - OCR-noisy (non-dictionary tokens);
  - specificity (gold "unspecified" vs specific);
  - non-leaf / drift.
- **Diagnostics:**
  - candidate recall@1/5/15/30;
  - ICD category accuracy;
  - SNOMED IS-A distance (needs the hierarchy; if no hierarchy, the open-ontology `is_a` from Mondo/DOID as a proxy);
  - ECE;
  - coverage–accuracy;
  - latency (median / p95 ms per term);
  - LLM calls per 1k terms.
- **Statistics:** paired bootstrap (1,000 resamples) on per-row correctness for every A/B. Ship only if the 95% CI excludes 0, or the gain is ≥ 1 pt and consistent across ≥ 4/5 folds.
- **Ablation table:** each component on/off. This feeds the write-up's "what was tried and abandoned".
- **Determinism:** run predict twice and `cmp`; fixed seeds; deterministic LightGBM; cached LLM responses.

## H. Experiment queue (ordered by expected gain ÷ cost; machine; owner)
| # | Experiment | Machine | Owner | Expected |
|---|---|---|---|---|
| 1 | Baseline + SapBERT base: `ranker_lab dump/run`, `evaluate --cv 5` | GPU box (running) | C | +1–4 pt recall |
| 2 | Data cleaning C1–C4 + medspaCy ConText | CPU | A | +1–2 pt |
| 3 | Local LLM (Qwen2.5-32B-AWQ, vLLM) adjudication, threshold 0.8, k = 15, vs Gemini | GPU | B | +8–12 pt (measured +9/+11 with Gemini) |
| 4 | Fine-tune SapBERT on training pairs + ontology synonyms, hard negatives (fold-safe) | GPU | A | +3–6 pt recall, +2–4 exact |
| 5 | Cross-encoder reranker (BiomedBERT init), hard negatives, feature into LambdaRank | GPU | B | +3–6 pt exact |
| 6 | Stage classifiers (ICD category, SNOMED tag, unspecified, abbreviation sense) | CPU/GPU | B | +1–3 pt |
| 7 | Cross-system joint pass + isotonic calibration + LLM routing on calibrated confidence | CPU | B | +1–2 pt, better abstention |
| 8 | BM25 + SymSpell + LLM query rewrite for recall | CPU/GPU | A | +1–2 pt recall |
| 9 | ICD tree-descent LLM variant; ensemble voting | GPU | B | ±, test |
| 10 | Error clustering, ablations, bootstrap CIs → write-up and demo | CPU | C | judging 15% + 15% |

## I. New or changed files (for implementation after approval)
- `termmap/clean.py`: label audit, chunk scaffolding stripper, sentence-level context (medspaCy).
- `termmap/classifiers.py`: OOF stage classifiers → features.
- `termmap/train_encoder.py`: SapBERT contrastive fine-tune (sentence-transformers MultipleNegativesRankingLoss + hard negatives); writes `models/sapbert-ft/`.
- `termmap/cross_encoder.py`: train and score the cross-encoder; feature `ce_score`.
- `termmap/calibrate.py`: isotonic calibration, routing threshold.
- `termmap/engine.py`: hooks for the new sources and features. Fold-safety: every learned component is trained only on the training folds inside `oof_features` / `fit`.
- `termmap/llm.py`: candidate orderings, majority vote, tree-descent prompt, vLLM (openai_compat) defaults.
- `scripts/evaluate.py`: slices, bootstrap CIs, ablation flags (`--disable sap,ce,llm…`).
- `scripts/bench.py`: latency and LLM-cost report.
- Docs: `docs/experiments.md` (live results table), `docs/architecture.md` update.

## Verification
Every experiment row is accepted only via `evaluate --cv 5` plus a paired bootstrap against the previous best, logged in `docs/experiments.md`. The final submission run must be deterministic (two runs, `cmp`) and run end to end from `scripts/predict.py`.

## Sources
- BELB benchmark: https://arxiv.org/pdf/2308.11537
- SapBERT: https://arxiv.org/abs/2010.11784
- LLM tree search for ICD: https://arxiv.org/pdf/2310.06552
- Reliable clinical coding / near misses: https://arxiv.org/html/2510.07629v1
- Few-shot clinical IE / CASI: https://arxiv.org/pdf/2205.12689
- Medical abbreviation disambiguation (MeDAL): https://www.biorxiv.org/content/10.1101/2025.09.12.675926.full.pdf
- Multilingual concept normalization (retrieve + rerank): https://www.medrxiv.org/content/10.1101/2025.01.15.25320579.full.pdf
- LLM mention simplification: https://tagteam.harvard.edu/hub_feeds/4354/feed_items/13797847
- BeLink (generative reranking): https://www.alphaxiv.org/abs/2605.22501
- SNOMED CT Entity Linking Challenge (JAMIA 2025): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12361850/
- OntolinkX thesis (SapBERT + cross-encoder): https://theses.gla.ac.uk/85080
- NegEx / ConText comparison: https://mitre.org/sites/default/files/publications/13-3434.pdf
- DEEPEN: https://scholarworks.indianapolis.iu.edu/items/2c3a73af-a037-4448-b777-d5ec58f10be6
- RRF: https://www.zeroentropy.dev/concepts/reciprocal-rank-fusion/
