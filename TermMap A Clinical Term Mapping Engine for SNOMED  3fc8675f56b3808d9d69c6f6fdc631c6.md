# TermMap: A Clinical Term Mapping Engine for SNOMED CT and ICD-10-CM

> Conducted By: HiLabs | Deadline: 18th October, 2026
> 

Submission link:  · Dataset link: [https://hiringdrive2026.blob.core.windows.net/hackathon/hackathon_train.csv](https://hiringdrive2026.blob.core.windows.net/hackathon/hackathon_train.csv)
Code System Description: 
ICD - [https://hiringdrive2026.blob.core.windows.net/hackathon/icd10cm_codes.xlsx](https://hiringdrive2026.blob.core.windows.net/hackathon/icd10cm_codes.xlsx)
SNOMED - [https://hiringdrive2026.blob.core.windows.net/hackathon/SNOMEDCT_US_codes_and_descriptions.xlsx](https://hiringdrive2026.blob.core.windows.net/hackathon/SNOMEDCT_US_codes_and_descriptions.xlsx)

## 🧭 Background

The clinical data world runs on unstructured text. Progress notes, discharge summaries, referral letters, scanned faxes, lab reports; the overwhelming majority of what a payer or provider actually holds about a member is prose written by a clinician under time pressure, full of abbreviations, misspellings, local shorthand and copy-forward noise.

HiLabs has spent years building tools that pull structured information out of that mess. CodEx, our clinical-context pipeline, reads a medical chart and returns the clinical entities it finds: problems, procedures, lab observations, medications, immunizations and vitals. That is the first half of the job.

The second half is coding, or in clinical language term mapping. An extracted string is not yet usable data. "HTN", "hypertension", "h/o high blood pressure" and "essential HTN, well controlled" are four different strings that all need to become I10 in ICD-10-CM and 59621000 in SNOMED CT before anything downstream can reason over them - quality measure logic, risk adjustment, cohort building, analytics, FHIR interchange. Until a term is coded, it is a string; once it is coded, it is data.

Today this step is the accuracy bottleneck. Existing approaches include deterministic dictionary lookup, then embedding-based retrieval, then reranking; get most of the way there, but the residual error concentrates exactly where it hurts: common conditions with many near-identical candidate codes, negated or historical mentions, laterality and acuity qualifiers, and long tail terms that appear in no dictionary. Every mis-mapped code propagates silently into whatever consumes it.

This challenge is about closing that gap. Build a term mapping engine that takes an extracted clinical entity and assigns the correct ICD-10-CM and SNOMED CT codes, at better than 95% accuracy, for diagnoses and conditions.

## 📚 The Coding Systems

Term Mapping layer handles nineteen canonical coding systems - RxNorm and NDC for drugs, LOINC for lab observations, CPT / HCPCS / ICD-10-PCS for procedures, CVX for immunizations, plus HL7 and FHIR value sets for acts, locations, participation and allergy classification. The full catalogue lives on the CDI space's Canonical Coding Systems page; DSM-5, NANDA-I and DRG are named there as systems we deliberately do not handle today.

This challenge takes two of those nineteen ICD10CM and SNOMEDCT_US because they carry diagnoses and conditions, and because they are where the residual accuracy gap is largest.

|  | ICD10CM | SNOMEDCT_US |
| --- | --- | --- |
| Full name | International Classification of Diseases, 10th Revision, Clinical Modification | Systematized Nomenclature of Medicine – Clinical Terms, U.S. Edition |
| What it is | The U.S. version of ICD-10, used to classify and code diagnoses | A comprehensive clinical vocabulary covering diagnoses, symptoms, procedures and findings |
| Used for | Diagnosis coding and reimbursement | Clinical documentation and decision support |
| Who uses it | Hospitals, outpatient clinics, insurers | EHR systems, health IT, clinicians |
| Example | E11.9 → Type 2 diabetes mellitus without complications; I10 → Essential (primary) hypertension | 44054006 → Diabetes mellitus type 2; 22298006 → Myocardial infarction |

### ICD-10-CM

The International Classification of Diseases, 10th Revision, Clinical Modification is the US billing and reporting standard for diagnoses, maintained by CDC/NCHS and updated annually (effective every October 1). It is a strict hierarchy: roughly 70,000+ billable codes arranged in chapters by body system and etiology, with codes of the form A00 to Z99 plus up to four further characters of specificity.

What makes it hard to map to:

- Specificity is mandatory and encoded in the code itself. S52.5 is not billable; S52.501A (unspecified fracture of lower end of right radius, initial encounter for closed fracture) is. Laterality, encounter type, and episode of care are all baked into trailing characters that the source text often does not state.
- "Unspecified" codes are legitimate targets. When the note says only "diabetes", E11.9 is the right answer, not a guess at a complication code. Knowing when not to be specific is half the skill.
- Excludes1/Excludes2 notes and combination codes. Some conditions that appear as two entities in text collapse into one code; others must never be coded together.

### SNOMED CT

SNOMED Clinical Terms is a poly-hierarchical ontology rather than a classification; maintained by SNOMED International, released twice yearly, with roughly 350,000+ active concepts in the international release plus a US extension. Each concept has a numeric ConceptId, one Fully Specified Name, a preferred term, and many synonyms; concepts are linked by IS-A relationships and attribute relationships (finding site, causative agent, associated morphology, severity).

What makes it hard to map to:

- Many valid answers at different levels of granularity. "Chest pain" could map to the general finding or to a more specific descendant; the correct target depends on convention, not just semantics.
- Synonym density. A single concept may carry a dozen descriptions, several of which collide with descriptions of neighbouring concepts. Pure lexical similarity is systematically misleading here.
- Post-coordination exists but is out of scope. We map to a single pre-coordinated concept.

### Why both, and why they disagree

ICD-10-CM is built for reimbursement and statistical reporting; SNOMED CT is built for clinical meaning. The two do not line up cleanly, a single SNOMED concept can correspond to several ICD-10-CM codes and vice versa. UMLS and the official SNOMED-to-ICD-10-CM map exist and are useful signal, but the map is incomplete, contains one-to-many and conditional targets, and cannot be treated as ground truth. A system that maps well to one system and derives the other from a crosswalk will not reach 95% on both.

## 🎯 Problem Statement

> "Build a term mapping engine that assigns the correct ICD-10-CM code and the correct SNOMED CT concept to a clinical term extracted from an unstructured medical chart, achieving greater than 95% mapping accuracy on diagnoses and conditions."
> 

The engine must be capable of:

- Mapping terms as they actually appear - abbreviations, misspellings, OCR noise, qualifier-laden phrases, and terms absent from any dictionary
- Disambiguating using the surrounding chunk context, not the term alone
- Calibrating specificity: choosing the unspecified code when the text supports nothing more, and the specific one when it does
- Predicting both coding systems without collapsing one into a crosswalk of the other
- Running end to end over the evaluation file without manual intervention

### In scope

- Diagnoses and conditions only. Entities of type Problem — what a medical coder identifies as a diagnosis, condition, sign or symptom.
- Both target systems. A submission must produce both an ICD-10-CM code and a SNOMED CT concept id for each input term.
- Single pre-coordinated code per system. One best code, not a ranked list, not a post-coordinated expression.
- Terms as they appear in real charts - abbreviations, misspellings, truncations, qualifier-laden phrases, and terms that have no exact dictionary entry.

### Out of scope

- Procedures, medications, labs, immunizations and vitals (CPT, HCPCS, RxNorm, LOINC, CVX). The same problem exists for each, but this challenge is deliberately narrowed.
- Entity extraction itself. Terms are given to you already extracted; you are not being asked to do NER.
- Anything requiring the full chart. You get the term and its chunk context, not the document.

### The bar

> 95% mapping accuracy on the held-out evaluation set, on both coding systems.
> 

This is a high bar and it is meant to be. It is roughly the level at which a mapping engine can be trusted in production without a human adjudication step behind it, which is the actual goal.

## 📂 The Datasets

### 1. Labelled dataset (training + evaluation)

Terms extracted by CodEx from de-identified medical charts, each with a human-adjudicated gold mapping.

| Column | Description |
| --- | --- |
| chunk_id | Identifier of the source chunk the term was extracted from. Provides context and allows grouping by document. |
| clinical_entity | The extracted term, verbatim as CodEx produced it; including abbreviations, casing and OCR noise. |
| icd10cm_code | Gold ICD-10-CM code. |
| snomed_code | Gold SNOMED CT concept id. |

[https://app.notion.com](https://app.notion.com)

### Notes on data use

- External public resources (UMLS, the official SNOMED→ICD-10-CM map, published ontologies, open-weight models) are permitted and encouraged, subject to their licences. If you use one, say so.
- The dictionaries define the legal answer space. A predicted code that is not in the supplied dictionary is scored as wrong.

## 🧠 Core Requirement: Accuracy on the Hard Cases

A string-similarity baseline will get you to the mid-80s in an afternoon. Everything after that is the actual problem. These are the failure modes that matter:

1. The nearest string is the wrong concept. "Hypertension" and "pulmonary hypertension" differ by one word and are completely different codes (I10 vs I27.20). "Malignant hypertension" is closer in character space to "hypertension" than "HTN" is, but "HTN" is the correct match. Lexical distance and clinical distance are not correlated in the region where errors happen.
2. Abbreviation collision. MS is multiple sclerosis, mitral stenosis, or morphine sulfate. RA is rheumatoid arthritis or right atrium. PID is pelvic inflammatory disease or prolapsed intervertebral disc. Disambiguation needs chunk context, and the context is often thin.
3. Specificity calibration. Given "diabetic neuropathy", the coder must decide between E11.40 and the family of more specific neuropathy codes — and must not invent specificity the text does not support. Over-coding and under-coding are both errors, and most systems are systematically biased toward one.
4. Granularity mismatch between systems. The correct SNOMED concept and the correct ICD-10-CM code are not in one-to-one correspondence. Deriving the second from the first via the official crosswalk imports the crosswalk's gaps and ambiguities.
5. The long tail. Rare conditions, eponyms, newly added codes, terms that exist in the chart but in no dictionary. The tail is where the last 5% lives, and it cannot be memorized from training data.
6. Noise from upstream. Real extracted terms carry OCR artifacts, truncation at chunk boundaries, and stray punctuation. "chronickidney disease stag 3" is a real kind of input, not a synthetic adversarial example.
7. Annual and biannual drift. ICD-10-CM changes every October, SNOMED twice a year. A solution that hardcodes a snapshot ages badly. How a submission handles version changes is a legitimate design consideration.

## 🛠️ Suggested Approaches

These are directions worth considering, not a prescribed architecture. A submission that ignores all of them and works is a better submission.

Normalization first. A lot of the gap is spelling, casing, punctuation and expansion. Abbreviation expansion conditioned on chunk context, before any retrieval, is cheap and moves the number.

Retrieve then rerank. Dense retrieval over concept descriptions (biomedical sentence encoders such as SapBERT or its successors) to pull a candidate set, then a stronger reranker that sees the term, the context and the full candidate descriptions. The retriever's job is recall; the reranker's job is the decision.

Don't discard lexical signal. Hybrid retrieval — BM25 alongside dense vectors — reliably beats either alone on this kind of task, because rare terms and exact code descriptions are precisely where embeddings are weak.

Exploit the hierarchy. Both systems are structured. Predicting the chapter or the three-character block first, then descending, turns one 70,000-way decision into a sequence of tractable ones — and makes specificity calibration an explicit choice rather than an accident.

Use an LLM where judgement is needed, not everywhere. LLMs are good at the disambiguation and specificity decisions and expensive as a first-pass retriever. A cascade — deterministic, then retrieval, then LLM adjudication only on low-confidence cases — is usually the right shape.

Couple the two systems. The official SNOMED→ICD-10-CM map is a strong prior even though it is not ground truth. Predicting both jointly, and using disagreement between the two predictions as a confidence signal, is unexplored territory here.

Mine the training data for what the dictionary lacks. Gold mappings encode local convention — how our coders resolve ambiguity. That convention is learnable and is not in any public resource.

## 🏆 Judging Criteria

| Criterion | Weight | What Judges Look For |
| --- | --- | --- |
| Mapping Accuracy | 50% | Exact-match accuracy on the held-out set, scored separately per system; the headline number is the lower of the two |
| Long-Tail Generalization | 20% | Accuracy on codes rare or absent in the training split — evidence the system reasons rather than memorizes |
| Engineering Quality & Reproducibility | 15% | Runs end to end from one command; deterministic; documented; deployable at chart volume |
| Error Analysis & Insight | 15% | The write-up shows real understanding of where it fails and why, with a credible plan for the remaining gap |

Proposed primary metric — exact-match accuracy per coding system. The fraction of evaluation terms for which the predicted code equals the gold code, computed separately for ICD-10-CM and SNOMED CT. The headline number is the lower of the two, so a system cannot win by being excellent at one and mediocre at the other.

Proposed secondary metrics, for diagnostics rather than ranking:

- Hierarchical / partial credit — for ICD-10-CM, credit for landing in the correct category or three-character block; for SNOMED, distance to the gold concept in the IS-A hierarchy. This separates "wrong concept" from "right concept, wrong specificity", which are very different failures operationally.
- Accuracy on the long tail — accuracy restricted to terms whose gold code appears rarely or not at all in the training split. This is the measure that predicts production behaviour on unseen charts.
- Coverage / abstention — if a system can decline to map, accuracy on the terms it does attempt, alongside how many it declined. A calibrated system that abstains on 3% and is 98% accurate on the rest may be more useful than one that guesses everything at 95%.
- Latency and footprint — median and p95 time per term, and whether the approach is deployable at chart-processing volume.

Open questions for the team to settle:

- Is there partial credit at all, or is the headline number strictly exact match?
- How are terms with genuinely ambiguous gold answers handled — excluded, or accepted against a set of valid codes?
- Is the evaluation set adjudicated by more than one annotator, and do we publish inter-annotator agreement as the practical ceiling?
- Are there per-category minimums (e.g. no coding chapter below some floor), to discourage optimizing for frequent conditions only?

## 📦 Deliverables

- A working mapping engine. Given a file of chunk_id, clinical_entity rows, it produces chunk_id, clinical_entity, icd10cm_code, snomed_code for every row. It must run end to end on the evaluation set without manual intervention.
- Predictions on the evaluation set, in the specified schema.
- Code, in a repository, with a README that a colleague can follow to reproduce the run.
- A short write-up (2–3 pages) covering: the approach and why; what was tried and abandoned; an error analysis of where it still fails and what you would do next; and any external resources or models used.
- A demo — a short walkthrough showing the engine mapping live input, including at least a few hard cases.

## 🧰 Suggested Tech Stack (not prescriptive)

| Component | Options |
| --- | --- |
| Lexical retrieval | Elasticsearch / OpenSearch, BM25 via rank_bm25, Lucene |
| Dense retrieval | SapBERT, BioLORD, MedCPT, E5 / BGE with domain fine-tuning |
| Vector index | FAISS, ScaNN, Qdrant, Chroma, pgvector |
| Reranking | Cross-encoder (MS MARCO / biomedical fine-tune), LLM reranker |
| LLM adjudication | See free-tier options below |
| Terminology tooling | UMLS Metathesaurus, pymedtermino, Snowstorm, SNOMED OWL toolkit, simple-icd-10-cm |
| String normalization | rapidfuzz, symspellpy, scispacy abbreviation detector |
| Data layer | Pandas, DuckDB, SQLite |
| Serving / demo | FastAPI, Streamlit, Gradio, or CLI |

### 💡 Free-tier LLM options

All three have free tiers with no billing required for basic usage:

- Google AI Studio (Gemini API) — aistudio.google.com; capped RPM/TPM, no billing needed
- OpenRouter — openrouter.ai; several :free model slugs available
- Mistral La Plateforme — console.mistral.ai; free experimentation tier

Self-hosted open-weight models are equally welcome, and may be the better fit here given the data cannot leave our environment — see Constraints.

## ⚖️ Constraints & Rules

- The engine must be deterministic and reproducible for a fixed input. If there is sampling anywhere, seeds are fixed and documented.
- No manual mapping of evaluation terms, no hand-curated overrides fitted to the evaluation set. Rules and dictionaries built from training data are fine; anything fitted to the held-out set is not.
- Any external model or dataset used must be nameable and licence-compatible.
- Reasonable inference cost. An approach that needs a large model call per term at chart volume should say so honestly in the write-up and quantify it.

Additionally:

- The dataset is proprietary clinical data. It must not be published, shared outside the challenge, posted to public repositories, or sent to any third-party API that retains data. Use private repos for submission.
- Never hardcode API keys or credentials. Use environment variables or a .env file, and list .env in .gitignore.
- External resources must be named in the write-up — every model, dataset, ontology release and crosswalk version.
- Both coding systems must be predicted. A submission that produces only one is incomplete, not partially scored.

## 📋 Submission Requirements

Every team must submit all three of the following:

### 1. 📄 Documentation PDF

A structured PDF covering:

- Approach — how the team interpreted the problem and scoped the solution
- Architecture — diagram of the mapping pipeline, its stages, and where each model or resource sits
- Design decisions — why this retrieval strategy, this reranker, this specificity policy
- Error analysis — where it still fails, broken down by failure mode, with examples
- Innovations — novel techniques or creative extensions beyond the core requirement
- Challenges & learnings — what didn't work, what you'd do differently

### 2. 💻 GitHub Repository (private, with access granted to judges)

- A clean, well-organized codebase
- A README.md covering: overview, setup, environment variables, and how to run the engine end to end
- A .env.example showing required keys (never commit real keys)
- A single command that reproduces the submitted predictions from the evaluation input file

### 3. 🎥 Demo Video (2–3 minutes, screen recording)

A screen-recorded walkthrough that:

- Explains the architecture briefly (30–45 seconds)
- Maps live input end to end, including several deliberately hard cases (an abbreviation collision, an OCR-mangled term, a specificity judgement call)
- Shows the error analysis or confidence output, not just the happy path

> Upload to an internal drive and include the link in the README.md and the PDF.
>