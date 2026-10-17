# Data analysis and answers to the open questions

Profiled 17 Oct 2026 from the files in this repo (training CSV in full; dictionaries loaded fully for stats).

## What the files contain

### `hackathon_train.csv` (9,646 rows)
Columns are `entity, chunk_text, coding_system, code, display, subconcept`. This is **not** the problem statement's `chunk_id, clinical_entity, icd10cm_code, snomed_code`. All text is lowercased.

| subconcept | rows | systems |
|---|---|---|
| DiagnosisCode | 5,005 | ICD 2,305 + SNOMED 2,700 |
| ProcedureCode | 2,778 | SNOMED (1 ICD) |
| LabService | 858 | SNOMED |
| EncounterType | 567 | SNOMED |
| Device | 437 | SNOMED |

- **Every entity string appears exactly once per coding system.** The set is de-duplicated by surface form, so evaluation entities will be unseen strings and exact-string memorisation is worth nothing.
- Diagnosis gold codes: ICD has 866 distinct codes (465 seen once); SNOMED has 1,669 distinct codes (1,199 seen once).
- Entity + chunk pairs: 2,123 are labelled in both systems, 576 only in SNOMED, 180 only in ICD.
- On an 80/20 split grouped by chunk, the validation gold code appears in training for **76% of ICD rows and 51% of SNOMED rows**. Everything above that has to come from dictionary retrieval, which is what the long-tail criterion measures.
- Exact match of entity to a dictionary description recovers the gold code for only **4% of ICD rows and 17% of SNOMED rows**.
- **Version drift:** 98.1% of ICD gold codes are leaves of the supplied dictionary. The non-leaf golds (`M51.36`, `I47.2`, `I47.1`, …) are codes that were split in FY2024/25, so the labels predate the dictionary. The dictionary itself is FY2026: `G35` is now split into `G35.A`/`G35.B*`/`G35.D`.
- Label noise: two rows have the invalid code `L6.0`.
- SNOMED gold semantic tags: disorder 1,276, finding 1,035, morphologic abnormality 72, situation 65, observable 52, procedure 34, …
- Within the training data the two systems are not 1:1. 146 of 1,325 SNOMED codes pair with more than one ICD code, and 286 of 824 ICD codes pair with more than one SNOMED code.
- Conventions differ between systems. For example, `lad` maps to ICD `R59.0` (enlarged lymph nodes) but to SNOMED `274621000` (*Lymphadenopathy absent*). Some ICD entities contain the code itself (`f31.9 bipolar disorder, unspecified`).
- `chunk_text` has a median length of 168 characters and a maximum of 23k. It is often a table dump (`header_1: :: header_2: …`), and the entity occurs in its chunk 99.9% of the time.

### `SNOMEDCT_US_codes_and_descriptions.xlsx`
984,585 description rows covering 382,124 concepts (the FSN with its semantic tag, plus synonyms). The Notes sheet says it is an export of HiLabs' existing Term Map index built on **gte_base_fp32**, which is the incumbent embedding model. It has no active flag and no hierarchy.

### `icd10cm_codes.xlsx`
159,306 rows covering 98,222 codes. That includes 111 range rows (`A00-B99`), category headers, and several descriptions per code. There is no billable flag, so leaves are derived from prefixes (74,718 leaves).

## Answers to the questions in the GPT pre-analysis
1. **Where is the chunk context?** In `chunk_text`; there is no `chunk_id`. We assume evaluation input is `entity, chunk_text`, and `termmap.data.read_input` accepts either schema.
2. **Size, distribution, repeats:** see above. There are no repeated entities and the tail is heavy.
3. **Split by rows, chunks or patients?** Unknown. We validate with a split grouped by `chunk_text` (`scripts/evaluate.py`).
4. **Do the dictionaries have synonyms and flags?** SNOMED has synonyms and FSNs but no status flags. ICD has multiple descriptions per code, no billable flag, and includes ranges and headers.
5. **Is 95% an absolute bar?** We can't know. We optimise min(ICD, SNOMED) and report long-tail accuracy and coverage-vs-accuracy.
6. **Can external models run locally?** Yes, the rules allow it. The cloud dev container blocks Hugging Face and NLM hosts until they are allowlisted.
7. **Scope:** train and evaluate on `DiagnosisCode`. Non-diagnosis SNOMED rows are used only as retrieval aliases.
8. **Output schema:** one row per input with `icd10cm_code` and `snomed_code`. Each system is predicted independently.
