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
| L1 | Gemini 2.5 Flash on conf < 0.6, top-8 (on #5, first 200 split terms) | split-200 | 0.520 → 0.610 | 0.545 → 0.655 | | | | | cloud |

CV5 run #6 extra diagnostics:
- **ICD 3-char category accuracy:** 0.758.
- **Accuracy at 97/90/80% coverage:** ICD 0.663 / 0.695 / 0.747; SNOMED 0.596 / 0.630 / 0.679.
- **Latency:** 22 / 26 ms per term.

Notes on #7: an ablation on the same features moved ICD −0.9 / SNOMED +0.9 points, which is noise. The polarity flag fires on 19–36% of terms, too broad to be discriminative. Drift-gold terms: 9 in the split, 1 correct with or without the feature. Kept as neutral and cheap; verdict pending CV5 with SapBERT. Conclusion: the remaining decision errors need semantic judgement (LLM / cross-encoder), not more hand-made features.

#7 CV5 verdict: no gain over #6 (ICD 0.6473 → 0.6469, SNOMED 0.5804 → 0.5752). The polarity features were removed. The retrieval-side additions (coronary abbreviations, drift-parent candidates) were kept because they're recall-neutral-to-positive.

L2: the local 14B LLM overriding all terms under 0.8 confidence hurt by 4–6 points. It replaced many ranker picks that were already right (terms with confidence 0.5–0.8 are 50–80% correct). Next: route by measured per-confidence-bucket accuracy (`ranker_lab llm` now prints this, plus policy comparisons), try 1 term per request (`TERMMAP_LLM_BATCH=1`), and try the 32B model.
