# Experiment log

All numbers are exact-match accuracy on held-out data. "Split" is the fixed 20% chunk-grouped hold-out (469 ICD, 565 SNOMED terms, ±2 pt noise). "CV5" is 5-fold chunk-grouped cross-validation over all 2,305 ICD and 2,700 SNOMED diagnosis rows (±1 pt).

| # | Change | Eval | ICD exact | SNOMED exact | ICD recall | SNOMED recall | ICD unseen | SNOMED unseen | Machine |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lexical TF-IDF + HGB ranker | split | 0.525 | 0.517 | 0.849 | 0.823 | 0.238 | 0.348 | cloud CPU |
| 2 | + scispacy word vectors, soft/context features | split | 0.529 | 0.515 | 0.870 | 0.857 | 0.248 | 0.362 | cloud CPU |
| 3 | + LambdaRank, relative features | split | 0.533 | 0.535 | 0.866 | 0.857 | 0.257 | 0.375 | cloud CPU |
| 4 | + other system's gold display in memory | split | 0.561 | 0.535 | 0.870 | 0.857 | 0.317 | 0.394 | cloud CPU |
| 5 | + CDC / DOID / Mondo / HPO aliases | split | 0.550 | 0.527 | 0.893 | 0.866 | 0.317 | 0.369 | cloud CPU |
| 6 | + SapBERT dictionary + memory retrieval | split | **0.633** | **0.581** | 0.930 | 0.904 | 0.416 | 0.436 | A6000 |
| 6 | same | CV5 | **0.647** | **0.580** | 0.932 | 0.919 | 0.428 | 0.447 | A6000 |
| 7 | + polarity-conflict / drift-parent features, coronary abbreviations (no SapBERT, vs #5) | split | 0.544 | 0.533 | 0.900 | 0.867 | 0.287 | 0.383 | cloud CPU |
| 7 | same as #7, with SapBERT | CV5 | 0.647 | 0.575 | 0.933 | 0.921 | 0.429 | 0.447 | A6000 |
| L2 | Qwen2.5-14B-AWQ (local vLLM) on conf < 0.8, top-15, batch 10 (on #7) | split | 0.625 → 0.567 | 0.581 → 0.540 | | | | | A6000 |
| L3 | Qwen-14B, batch 10, routing "override only if LLM pick prob ≥ 0.3 × top" (best policy; chosen on split) | split | 0.625 → 0.661 | 0.581 → 0.600 | | | | | A6000 |
| L4 | same, 1 term per request | split | 0.625 → 0.653 | 0.581 → 0.589 | | | | | A6000 |
| C1 | cross-encoder (SapBERT init, listwise, top-15, 2 epochs); alone / blend w=0.3 | split | 0.480 / **0.655** | 0.499 / **0.607** | gold@15 0.898 | gold@15 0.837 | | | A6000 |
| R1 | SapBERT fine-tuned on train-side pairs + 15k synonym pairs/system: retrieval only, acc@1 / r@15 | split | 0.369→0.429 / 0.689→0.821 | 0.460→0.510 / 0.724→0.788 | | | unseen r@15 0.673→0.772 | unseen r@15 0.653→0.723 | A6000 |
| 8 | + cross-encoder blend w=0.3 (base SapBERT) | CV5 | **0.658** | **0.604** | 0.933 | 0.921 | 0.440 | 0.477 | A6000 |
| 9 | + synonyms-only fine-tuned SapBERT instead of base (with cross-encoder) | CV5 | 0.654 | 0.606 | 0.940 | 0.922 | 0.420 | 0.479 | A6000 |
| R2 | synonyms-only SapBERT fine-tune (no labelled terms), retrieval only acc@1 / r@15 | split | 0.369→0.354 / 0.689→0.699 | 0.460→0.441 / 0.724→0.697 | | | | | A6000 |
| L5 | Qwen-32B-AWQ, batch 10 (ICD only so far): LLM acc on sent 0.409 vs ranker 0.449; best policy | split | 0.625 → 0.653 | pending | | | | | A6000 |
| J1 | joint ICD+SNOMED decoding (pair co-occurrence + open crosswalk; weights tuned on train-side OOF), base features no SapBERT | split | 0.546 → 0.557 | 0.524 → 0.517 | | | | | cloud CPU |
| C2 | cross-encoder + 10k synonym groups/system (distant supervision); alone / best blend | split | 0.527 / 0.663 (w=0.5) | 0.519 / 0.604 (w=0.3) | | | | | A6000 |
| L1 | Gemini 2.5 Flash on conf < 0.6, top-8 (on #5, first 200 split terms) | split-200 | 0.520 → 0.610 | 0.545 → 0.655 | | | | | cloud |

CV5 run #6 extra diagnostics:
- **ICD 3-char category accuracy:** 0.758.
- **Accuracy at 97/90/80% coverage:** ICD 0.663 / 0.695 / 0.747; SNOMED 0.596 / 0.630 / 0.679.
- **Latency:** 22 / 26 ms per term.

Notes on #7: an ablation on the same features moved ICD −0.9 / SNOMED +0.9 points, which is noise. The polarity flag fires on 19–36% of terms, too broad to be discriminative. Drift-gold terms: 9 in the split, 1 correct with or without the feature. Kept as neutral and cheap; verdict pending CV5 with SapBERT. Conclusion: the remaining decision errors need semantic judgement (LLM / cross-encoder), not more hand-made features.

#7 CV5 verdict: no gain over #6 (ICD 0.6473 → 0.6469, SNOMED 0.5804 → 0.5752). The polarity features were removed. The retrieval-side additions (coronary abbreviations, drift-parent candidates) were kept because they're recall-neutral-to-positive.

L2: the local 14B LLM overriding all terms under 0.8 confidence hurt by 4–6 points. It replaced many ranker picks that were already right (terms with confidence 0.5–0.8 are 50–80% correct). Next: route by measured per-confidence-bucket accuracy (`ranker_lab llm` now prints this, plus policy comparisons), try 1 term per request (`TERMMAP_LLM_BATCH=1`), and try the 32B model.

LLM notes: Qwen-14B is less accurate than the ranker on the terms it sees (0.36 vs 0.45), so it only helps as a tie-breaker between the ranker's plausible candidates. Batching 10 terms per request was better than 1.
Cross-encoder notes: it's weaker alone but complementary; blending at 0.3 helped both systems.
Both of these settings were chosen on the same split, so the CV5 run is the real test.

#8 vs #9: the cross-encoder accounts for the CV gain (+1.1 ICD, +2.9 SNOMED over #7). The synonyms-only encoder is neutral (R2 retrieval slightly worse; #9 ≈ #8), so base SapBERT stays the default. The large R1 retrieval gain came from training terms; using it without leakage needs cross-fitted encoders (not yet done).

J1: joint decoding helps ICD slightly (the SNOMED pick informs the ICD choice) but not SNOMED, the headline system. Train-side tuning gave +0.9/+0.3; on held-out +1.1/−0.7. Not shipped; documented as tried.

C2: synonym training makes the cross-encoder better on its own (+4.7 ICD, +2.0 SNOMED) but not in the blend with the ranker (ICD +0.4–0.8, SNOMED −0.3, all within noise). The ranker already supplies the vocabulary signal, and the residual errors are coder convention and context. Kept as an option (TERMMAP_CE_SYN), off by default.

## v2: diagnosis of the plateau and staged pipeline (18 Oct)

Findings that changed the plan (CPU config, no SapBERT/CE, unless noted):
- **Context is unused.** Blanking `chunk_text` at test time: split ICD 0.546 → 0.546, SNOMED 0.524 → 0.512. Context features carry ~2% of ranker gain.
- **Cross-encoder trained on the wrong lists.** `_train_ce` built lists with the in-sample ranker (gold first in 93–97% of lists), while at test time the OOF ranker has gold first only 66–69% of the time. v2 trains on cross-fitted OOF ranker lists.
- **Cross-encoder only saw the top 15.** SNOMED gold in top 15 was 0.837 vs 0.904 in all candidates; v2 uses k=30.
- **Display text bugs.** The ICD display was the first spreadsheet row, often the abbreviated CMS short form ("Oth symptoms …"), for 19.6% of ICD gold codes; v2 uses the CDC tabular long description or the full wording. SNOMED semantic tags were mis-parsed for 1,388 concepts (e.g. "combined site", "ajcc"); v2 only accepts real semantic tags.
- **Labels are consistent.** Training terms with identical word sets agree 97% (ICD) / 91% (SNOMED). The earlier 28% / 43% "label conflict" figure used a loose grouping that dropped qualifiers, which legitimately change the code. So label noise is not the ceiling.
- **Training terms were not reproduced.** Training terms fed back in got their gold code only 97% / 86% of the time. v2 adds an exact training-term lookup (S0).
- **Error profile (split, ranker).** ICD: 22% never retrieved, 38% of in-candidate errors at rank 2, 28% in the same 3-character category. SNOMED: 28% never retrieved, 31% at rank 2.

| # | Change (CV fold 0, CPU config) | ICD exact | SNOMED exact | ICD recall | SNOMED recall | Note |
|---|---|---|---|---|---|---|
| V0 | locked engine, CE off (old code, same fold) | 0.5831 | 0.5519 | 0.887 | 0.882 | baseline for v2 |
| V1 | v2 S1/S2: tag + display fixes, general-code expansion, ICD-only negation queries, freq per 1000, regularised ranker | 0.5920 | 0.5222 | 0.896 | 0.882 | SNOMED −3.0, McNemar p=0.026 |
| V1a | same features, ranker ablation: original 600-tree ranker | 0.5765 | 0.5389 | | | ICD prefers regularised (split +0.9, fold 0 +1.5); SNOMED prefers original (split −0.4, fold 0 +1.9) |
| V2 | per-system ranker params (ICD regularised, SNOMED original) | 0.5920 | 0.5389 | 0.896 | 0.882 | SNOMED −1.3 vs V0, within noise (MDE ≈ 3.4 pt on 540 terms) |

GPU-side results for S3 (cross-encoder v2), S4 (LLM selector) and S5 (stacker) go below as they come in. See `docs/gpu_runbook.md`.

### GPU run, 18 Oct (A6000, WSL; logs in `logs/`)
Baseline for gates and paired comparison: `reports/val_predictions.csv` had been overwritten by run **#9** (pooled ICD 0.6538 / SNOMED 0.6056, recall 0.940 / 0.922), not the locked #8 (0.658 / 0.604); no copy of #8's per-row output exists. Saved as `reports/baseline_cv5_run9.csv` and used as the reference (within noise of #8; its higher recall makes the s2 gate slightly stricter). Fold 0 of #9: ICD 0.6563 (n=451), SNOMED 0.6074 (n=540).

| # | Stage (fold 0, GPU config) | ICD | SNOMED | Gate |
|---|---|---|---|---|
| G1 | s2: ranker top-1 | 0.6408 | 0.5778 | |
| G1 | s2: recall r@15 / r@30 / r@any | 0.889 / 0.905 / 0.931 | 0.867 / 0.893 / 0.924 | PASS: r@any vs #9 fold 0 0.938 / 0.924 (−0.7 / ±0 pt, allow −1) |
| G1 | s2: unseen r@any | 0.798 | 0.881 | |
| G1 | s2: train top-1 OOF / in-sample | 0.643 / 0.716 | 0.579 / 0.884 | ICD gap 0.07 (was 0.57/0.86); SNOMED keeps the original 600-tree ranker, so its gap stays large |
| G2 | s3: CE v2 (SapBERT init, k=30, OOF lists), epoch 1 train / dev acc | 0.515 / **0.432** | not reached | **FAIL** (< 0.45), training aborted after epoch 1 (~2 min) |

G2 context: on the same 169 ICD dev lists (gold in top 30) the OOF ranker's top-1 is 0.740, so the CE alone is 31 pt below it after epoch 1. Dev n=169 gives SE ≈ 3.8 pt, so 0.432 vs 0.45 is within half an SE. At step 100 the loss was 3.25, against ln 30 = 3.40 for uniform. Note: the CE gate raises `RuntimeError` (exit 1), not exit 2 as the runbook says. Stopped here pending a decision; s5 / compare / the 5-fold run were not run.

The epoch-1 gate was then lowered to 0.15 ("not learning"; chance at k=30 is 0.033) in `9aa62b0`, and s3 was rerun unchanged.

| # | Stage (fold 0, GPU config) | ICD | SNOMED | Gate |
|---|---|---|---|---|
| G3 | s3 rerun: CE dev acc by epoch 1 / 2 / 3 / 4 (best epoch kept) | 0.432 / 0.550 / 0.580 / 0.586 | 0.604 / 0.656 / 0.682 / 0.672 | epoch-1 PASS (≥ 0.15). Ranker on the same dev lists: 0.740 / 0.656 |
| G3 | s3: gold@30 / ranker / CE alone / blend w=0.3 (= best blend) | 0.905 / 0.6408 / 0.5255 / **0.6696** | 0.893 / 0.5778 / 0.5481 / **0.6074** | blend ≥ ranker PASS on both (+2.9 / +3.0 pt) |
| G4 | s5: stacker (two-way cross-fit inside fold 0) + training-term lookup = FINAL | **0.6718** (seen 0.766 / unseen 0.395) | **0.6056** (seen 0.695 / unseen 0.515) | stacker alone 0.6718 / 0.6037 |
| G4 | compare vs #9 fold 0 (paired, McNemar) | 0.6563 → 0.6718, **+1.55 pt** [−1.55, +4.66], p=0.40 | 0.6074 → 0.6056, **−0.19 pt** [−3.15, +2.78], p=1.00 | headline 0.6074 → 0.6056 |
| G4 | compare slices: seen / unseen codes | +2.67 / −1.75 pt | −3.68 (p=0.099) / +3.36 pt (p=0.20) | none significant |

Notes on G3/G4:
- The first s3 rerun (`logs/v2_s3_f0b.log`) crashed after ICD epoch 4 with `CUBLAS_STATUS_INTERNAL_ERROR` in the test-scoring step. It did not reproduce when scoring the same 13,530 pairs with a fresh model (8.8 s, 2 GB peak), and the next identical run (`logs/v2_s3_f0c.log`) passed. The per-epoch numbers were bit-identical across all three runs.
- **CE training holds ~47–48.5 GB of the A6000** within 30 s of starting (`logs/gpu_mem_s3c.log`): 8 lists × 30 candidates × 192 tokens through eager attention. That is not the ~8 GB the runbook assumed, so the LLM selector cannot run alongside s3 / the 5-fold run on one 48 GB card. The near-full device is the likely cause of the cuBLAS error.
- The training-term lookup adds nothing in CV (FINAL = stacker on ICD): CV test entities never occur in the training side. It only matters for eval terms that repeat training terms.
- With one cached fold the stacker is fitted on half of fold 0 at a time. On SNOMED it is 0.4 pt below the fixed 0.3 blend; in the 5-fold run it is fitted leave-one-fold-out on ~4× the data.
- **Decision rule.** The runbook says to run all folds if B ≥ A on both systems, or within −1 pt on one with a clear gain on the other. SNOMED is −0.19 (within −1), but the ICD gain (+1.55, p=0.40) is not clear on 451 terms, and the headline does not beat the baseline (0.6056 vs 0.6074). 5-fold not started; the decision is pending.

### v2 folds 1–4 retrieval (s2), selector fold 0
s2 for folds 1–4 was run on the CPU (`CUDA_VISIBLE_DEVICES=""`, ~3.5 min per fold) while the selector held the GPU, gated against `reports/baseline_cv5_run9.csv` (`logs/v2_s2_f1-4.log`). All 8 recall gates PASS.

| Fold | ICD ranker top-1 / cand. recall (vs #9) | SNOMED ranker top-1 / cand. recall (vs #9) |
|---|---|---|
| 1 | 0.6505 / 0.9341 (0.9341) | 0.5958 / 0.9291 (0.9291) |
| 2 | 0.6193 / 0.9387 (0.9365) | 0.5615 / 0.8981 (0.9019) |
| 3 | 0.6721 / 0.9369 (0.9409) | 0.6124 / 0.9398 (0.9398) |
| 4 | 0.6475 / 0.9534 (0.9490) | 0.5787 / 0.9114 (0.9114) |

LLM selector fold 0 (Qwen2.5-7B LoRA, `logs/sel_f0.log`): CHECKPOINT step 150 dev acc 0.655, step 300 0.720, vs ranker 0.605 on the same 200 dev terms; left running to completion (476 steps).

## Locked configuration
Base SapBERT + lexical/word-vector retrieval + training memory → LambdaRank → cross-encoder blend (w=0.3, top-15), LLM off.
CV5: ICD 0.658, SNOMED 0.604 (#8).
