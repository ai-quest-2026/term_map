"""Text normalization, abbreviation expansion and qualifier extraction."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^a-z0-9.+%\s]")
_SPACE_RE = re.compile(r"\s+")
# ICD-10-CM code written inside the entity, e.g. "f31.9 bipolar disorder, unspecified".
ICD_CODE_RE = re.compile(r"\b([a-tv-z][0-9][0-9a-z](?:\.[0-9a-z]{1,4})?)\b", re.I)

# Common clinical abbreviations (public clinical shorthand, not fitted to any evaluation data).
# Ambiguous ones list several expansions; the reranker uses chunk context to choose.
ABBREVIATIONS: dict[str, list[str]] = {
    "htn": ["hypertension"], "dm": ["diabetes mellitus"], "dm1": ["type 1 diabetes mellitus"],
    "dm2": ["type 2 diabetes mellitus"], "t2dm": ["type 2 diabetes mellitus"], "dmt2": ["type 2 diabetes mellitus"],
    "t1dm": ["type 1 diabetes mellitus"], "iddm": ["type 1 diabetes mellitus"], "niddm": ["type 2 diabetes mellitus"],
    "cad": ["coronary artery disease"], "chf": ["congestive heart failure"], "hf": ["heart failure"],
    "hfref": ["heart failure with reduced ejection fraction"], "hfpef": ["heart failure with preserved ejection fraction"],
    "ckd": ["chronic kidney disease"], "esrd": ["end stage renal disease"], "aki": ["acute kidney injury"],
    "arf": ["acute renal failure", "acute respiratory failure"], "copd": ["chronic obstructive pulmonary disease"],
    "afib": ["atrial fibrillation"], "af": ["atrial fibrillation"], "a fib": ["atrial fibrillation"],
    "aflutter": ["atrial flutter"], "afl": ["atrial flutter"], "mi": ["myocardial infarction", "mitral insufficiency"],
    "stemi": ["st elevation myocardial infarction"], "nstemi": ["non st elevation myocardial infarction"],
    "cva": ["cerebrovascular accident"], "tia": ["transient ischemic attack"], "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism", "pedal edema"], "gerd": ["gastroesophageal reflux disease"], "uti": ["urinary tract infection"],
    "uri": ["upper respiratory infection"], "bph": ["benign prostatic hyperplasia"], "osa": ["obstructive sleep apnea"],
    "oa": ["osteoarthritis"], "ra": ["rheumatoid arthritis", "right atrium"], "ms": ["multiple sclerosis", "mitral stenosis"],
    "pid": ["pelvic inflammatory disease", "prolapsed intervertebral disc"], "as": ["aortic stenosis", "ankylosing spondylitis"],
    "ar": ["aortic regurgitation", "allergic rhinitis"], "mr": ["mitral regurgitation"], "tr": ["tricuspid regurgitation"],
    "hld": ["hyperlipidemia"], "hl": ["hyperlipidemia"], "hlp": ["hyperlipidemia"], "hchol": ["hypercholesterolemia"],
    "aaa": ["abdominal aortic aneurysm"], "ashd": ["atherosclerotic heart disease"], "pvd": ["peripheral vascular disease"],
    "pad": ["peripheral arterial disease"], "sob": ["shortness of breath"], "doe": ["dyspnea on exertion"],
    "cp": ["chest pain"], "n/v": ["nausea and vomiting"], "nv": ["nausea and vomiting"], "lbp": ["low back pain"],
    "ha": ["headache"], "loc": ["loss of consciousness"], "ams": ["altered mental status"], "lad": ["lymphadenopathy"],
    "lft": ["liver function test"], "dka": ["diabetic ketoacidosis"], "gdm": ["gestational diabetes mellitus"],
    "brbpr": ["bright red blood per rectum"], "brpr": ["bright red blood per rectum"], "gi bleed": ["gastrointestinal hemorrhage"],
    "gib": ["gastrointestinal hemorrhage"], "ibs": ["irritable bowel syndrome"], "ibd": ["inflammatory bowel disease"],
    "nafld": ["nonalcoholic fatty liver disease"], "mdd": ["major depressive disorder"], "gad": ["generalized anxiety disorder"],
    "ptsd": ["post traumatic stress disorder"], "adhd": ["attention deficit hyperactivity disorder"], "ocd": ["obsessive compulsive disorder"],
    "bpd": ["bipolar disorder", "borderline personality disorder"], "etoh": ["alcohol"], "aud": ["alcohol use disorder"],
    "oud": ["opioid use disorder"], "sz": ["seizure"], "hx": ["history of"], "h/o": ["history of"], "s/p": ["status post"],
    "fx": ["fracture"], "ddd": ["degenerative disc disease"], "djd": ["degenerative joint disease"], "ak": ["actinic keratosis"],
    "sk": ["seborrheic keratosis"], "bcc": ["basal cell carcinoma"], "scc": ["squamous cell carcinoma"], "ca": ["cancer"],
    "le": ["lower extremity"], "ue": ["upper extremity"], "lle": ["left lower extremity"], "rle": ["right lower extremity"],
    "ble": ["bilateral lower extremity"], "lue": ["left upper extremity"], "rue": ["right upper extremity"],
    "rlq": ["right lower quadrant"], "llq": ["left lower quadrant"], "ruq": ["right upper quadrant"], "luq": ["left upper quadrant"],
    "lvh": ["left ventricular hypertrophy"], "rvh": ["right ventricular hypertrophy"], "lbbb": ["left bundle branch block"],
    "rbbb": ["right bundle branch block"], "svt": ["supraventricular tachycardia"], "vt": ["ventricular tachycardia"],
    "sss": ["sick sinus syndrome"], "pna": ["pneumonia"], "cap": ["community acquired pneumonia"], "ards": ["acute respiratory distress syndrome"],
    "ild": ["interstitial lung disease"], "ckd3": ["chronic kidney disease stage 3"], "ugib": ["upper gastrointestinal hemorrhage"],
    "lgib": ["lower gastrointestinal hemorrhage"], "sbo": ["small bowel obstruction"], "hcc": ["hepatocellular carcinoma"],
    "hsv": ["herpes simplex"], "hiv": ["human immunodeficiency virus infection"], "hcv": ["hepatitis c"], "hbv": ["hepatitis b"],
    "ra/oa": ["rheumatoid arthritis", "osteoarthritis"], "ckd/esrd": ["chronic kidney disease"], "vit d def": ["vitamin d deficiency"],
    "ida": ["iron deficiency anemia"], "itp": ["immune thrombocytopenic purpura"], "dvt/pe": ["deep vein thrombosis"],
    "ln": ["lymph node"], "oe": ["otitis externa"], "om": ["otitis media"], "uc": ["ulcerative colitis"], "cd": ["crohn disease"],
    "nash": ["nonalcoholic steatohepatitis"], "tbi": ["traumatic brain injury"], "sah": ["subarachnoid hemorrhage"],
    "ich": ["intracerebral hemorrhage"], "sdh": ["subdural hemorrhage"], "pud": ["peptic ulcer disease"], "abd": ["abdominal"],
    "w/": ["with"], "w/o": ["without"], "b/l": ["bilateral"], "bilat": ["bilateral"], "lt": ["left"], "rt": ["right"],
    "chr": ["chronic"], "ac": ["acute"], "neg": ["negative"], "pos": ["positive"], "hyperchol": ["hypercholesterolemia"],
    "hypothyr": ["hypothyroidism"], "hyperthyr": ["hyperthyroidism"], "anx": ["anxiety"], "dep": ["depression"],
    # Coronary anatomy shorthand from cath / echo reports ("30% focal mrca stenosis", "90% ladd1").
    "rca": ["right coronary artery"], "mrca": ["mid right coronary artery"], "prca": ["proximal right coronary artery"],
    "drca": ["distal right coronary artery"], "lcx": ["left circumflex coronary artery"],
    "lm": ["left main coronary artery"], "lmca": ["left main coronary artery"], "mlad": ["mid left anterior descending artery"],
    "plad": ["proximal left anterior descending artery"], "dlad": ["distal left anterior descending artery"],
    "ladd1": ["first diagonal branch of left anterior descending artery"], "d1": ["first diagonal branch"],
    "om": ["obtuse marginal branch", "otitis media"], "om1": ["first obtuse marginal branch"],
    "pda": ["posterior descending artery", "patent ductus arteriosus"], "lv": ["left ventricle"], "rv": ["right ventricle"],
    "lvef": ["left ventricular ejection fraction"], "bp": ["blood pressure"], "hr": ["heart rate"],
    "v/d": ["vomiting and diarrhea"], "n/v/d": ["nausea vomiting and diarrhea"], "tb": ["tuberculosis", "theraband"],
    "hcc": ["hierarchical condition category", "hepatocellular carcinoma"],
}
ABBREVIATIONS["lad"] = ["lymphadenopathy", "left anterior descending artery"]

_LATERALITY = {
    "left": "left", "lt": "left", "right": "right", "rt": "right", "bilateral": "bilateral", "b/l": "bilateral",
    "bilat": "bilateral", "both": "bilateral",
}
_NEGATION_RE = re.compile(r"\b(no|not|denies|denied|without|negative for|absent|neg|free of|ruled out|r/o)\b")
_HISTORY_RE = re.compile(r"\b(h/o|hx|history of|s/p|status post|prior|past|previous|remote)\b")


def basic(text: str) -> str:
    """Lowercase, unify separators, drop stray punctuation, collapse whitespace."""
    t = str(text).lower()
    t = t.replace("/", " / ").replace("-", " ").replace("_", " ")
    t = re.sub(r"(?<=[a-z])\.(?=\s|$)", " ", t)  # trailing dots of abbreviations
    t = _PUNCT_RE.sub(" ", t)
    t = t.replace(" / ", " ")
    return _SPACE_RE.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return basic(text).split()


def expansions(entity: str) -> list[str]:
    """Return alternative query strings: the normalized entity plus abbreviation-expanded variants."""
    raw = str(entity).lower().strip()
    norm = basic(raw)
    out = [norm]
    if raw in ABBREVIATIONS:
        out += ABBREVIATIONS[raw]
    if norm in ABBREVIATIONS:
        out += ABBREVIATIONS[norm]
    toks = norm.split()
    if any(t in ABBREVIATIONS for t in toks):
        # Expand every abbreviated token with its first (most common) reading,
        # plus one variant per alternative reading of ambiguous tokens.
        base = [ABBREVIATIONS[t][0] if t in ABBREVIATIONS else t for t in toks]
        out.append(" ".join(base))
        for i, t in enumerate(toks):
            for alt in ABBREVIATIONS.get(t, [])[1:]:
                out.append(" ".join(base[:i] + [alt] + base[i + 1:]))
    seen, uniq = set(), []
    for q in out:
        q = basic(q)
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


_POL_CUES = {"no", "not", "non", "without", "neither", "nor", "absent"}
_POL_STOP = {"of", "or", "and", "the", "a", "an", "any", "to", "with", "in", "on"}


def polarity(text: str) -> tuple[set[str], set[str]]:
    """(negated words, asserted words) inside one phrase: "acute non recurrent sinusitis" negates
    "recurrent"; "nonintractable" negates "intractable"; "without hypoxia" negates "hypoxia"."""
    toks = basic(text).split()
    neg: set[str] = set()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("non") and len(t) > 6 and t not in ("nonspecific",):
            neg.add(t[3:])
        if t in _POL_CUES or (t == "w" and i + 1 < len(toks) and toks[i + 1] == "o"):
            # "non" binds to the next word only ("non recurrent ethmoidal"); other cues to the next two.
            j, taken, scope = i + (2 if t == "w" else 1), 0, 1 if t == "non" else 2
            while j < len(toks) and taken < scope:
                if toks[j] not in _POL_STOP:
                    neg.add(toks[j])
                    taken += 1
                j += 1
        i += 1
    pos = {t for t in toks if t not in neg and t not in _POL_CUES and t not in _POL_STOP and not t.startswith("non")
           and len(t) > 1}
    return neg, pos


def icd_codes_in_text(entity: str) -> list[str]:
    return [m.upper() for m in ICD_CODE_RE.findall(str(entity))]


def laterality(text: str) -> str:
    found = {_LATERALITY[t] for t in str(text).lower().replace("/", " ").split() if t in _LATERALITY}
    if "bilateral" in found or {"left", "right"} <= found:
        return "bilateral"
    return found.pop() if found else ""


def is_negated(entity: str, chunk: str) -> bool:
    """Crude NegEx: a negation cue within the 6 tokens before the entity in its chunk."""
    e, c = str(entity).lower(), str(chunk).lower()
    i = c.find(e)
    if i < 0:
        return bool(_NEGATION_RE.search(e))
    window = " ".join(c[:i].split()[-6:])
    return bool(_NEGATION_RE.search(window)) or bool(_NEGATION_RE.search(e))


def is_history(entity: str, chunk: str) -> bool:
    e, c = str(entity).lower(), str(chunk).lower()
    i = c.find(e)
    window = " ".join(c[:max(i, 0)].split()[-4:]) if i >= 0 else ""
    return bool(_HISTORY_RE.search(e)) or bool(_HISTORY_RE.search(window))


def context_window(entity: str, chunk: str, width: int = 200) -> str:
    """The part of the chunk around the entity (chunks can be 20k chars of table dump)."""
    e, c = str(entity).lower(), str(chunk).lower()
    i = c.find(e)
    if i < 0:
        return c[: 2 * width]
    return c[max(0, i - width): i + len(e) + width]
