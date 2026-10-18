# Demo video script (2:30–2:50, screen recording)

The brief asks for: the architecture in 30–45 s; live input mapped end to end, including an abbreviation collision, an OCR-mangled term and a specificity call; and the confidence or error-analysis output, not just the happy path.

**Recording setup (Windows + WSL):**
- Use OBS Studio (Display Capture, 1920×1080, 30 fps, MP4) or Xbox Game Bar (Win+Alt+R). Trim in Clipchamp.
- Use the Streamlit app (`streamlit run app/streamlit_app.py`, http://localhost:8501), or a terminal with a font of at least 16 pt.
- Have the PDF open on page 1 (the architecture diagram).

**Before recording, map one preset once,** so the engine and the 7B selector are already loaded when you record.

| Time | On screen | Narration (key points) |
|---|---|---|
| 0:00–0:10 | PDF title | "TermMap maps a clinical mention plus its chart text to one ICD-10-CM and one SNOMED CT code, both from the supplied dictionaries. Everything runs locally and deterministically." |
| 0:10–0:45 | Architecture diagram | "We retrieve about 80 candidates per system from six sources: character and word TF-IDF for typos, SapBERT for synonyms, the training examples, and open ontologies. A LambdaRank ranker trained out-of-fold orders them. A cross-encoder reads the mention inside its context. A fine-tuned local Qwen 7B selector picks among the top 20. A stacker combines everything into a code and a calibrated confidence." |
| 0:45–1:50 | App tab "Map a term" with the presets | Run each case and point it out:<br>– **"ms":** multiple sclerosis in the neurology note, mitral stenosis in the echo note (abbreviation collision resolved by context).<br>– **"chronickidney disease stag 3":** CKD stage 3 despite the OCR damage.<br>– **"obesity class ii":** the specificity call; show the runner-up and the margin.<br>– **"acute infarct":** cerebral in the MRI note, myocardial in the troponin note.<br>– **Negated case:** SNOMED's "No focal neurology" concept, and a "route to review" flag on a low-confidence case. |
| 1:50–2:30 | App tab "Blind test" (100 terms) or "Batch + evaluate", then PDF page 2 | "Answers hidden, an engine trained without these terms: exact match per system, the headline, the long tail, accuracy when only the most confident terms are answered, and the error breakdown. Most remaining errors are choices between close siblings: the gold code is in the candidate list 92% / 89% of the time." |
| 2:30–2:45 | README "Reproduce" section | "One command reproduces the submitted predictions, and two runs are byte-identical. Fold-0 headline 0.685; 5-fold 0.622 without the selector." |

**If time is short, cut** the README shot. Keep the architecture, live run, hard cases and confidence.

**Before saying any expected answer above on camera, check it against what the app actually shows.** If the engine gets a case wrong, show it as an error-analysis example ("here it picks the specific code; the gold convention is unspecified"). That counts as "not just the happy path".
