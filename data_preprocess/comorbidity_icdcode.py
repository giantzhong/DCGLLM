"""
Centralized ICD code/regex definitions for comorbidity tasks.

We keep *regex strings* here (not compiled patterns) so that each dataset
script can compile them with its own flags (e.g., re.IGNORECASE).
"""

from __future__ import annotations


# Hypertension (HTN): ICD-9/10 prefixes
HTN_REGEX = r"^(401|402|403|404|405|I10|I11|I12|I13|I15)"

# Diabetes Mellitus (DM): ICD-9/10 prefixes
DM_REGEX = r"^(250|E0[8-9]|E1[0-3])"

# Cardiovascular Disease (CVD) strict regex (from data_view.ipynb L13-L27)
CVD_STRICT_REGEX = (
    r"^(?:"
    r"41[0-4]"          # ICD-9: 410–414 (MI / IHD / CHD)
    r"|428"             # ICD-9: HF
    r"|43[0-8]"         # ICD-9: 430–438 (Stroke / Cerebrovascular)
    r"|4273[12]"        # ICD-9: AF / AFlutter
    r"|4402"            # ICD-9: PAD strict = 440.2*
    r"|I2[0-5]"         # ICD-10: I20–I25 (IHD / MI family)
    r"|I50"             # ICD-10: HF
    r"|I6[0-9]"         # ICD-10: I60–I69 (Stroke)
    r"|I48"             # ICD-10: AF
    r"|I70[2-7]"        # ICD-10: PAD I70.2–I70.7
    r"|I739"            # ICD-10: PAD I73.9
    r")"
)

# Chronic Kidney Disease (CKD) strict regex
# Per request: only use these two strict prefixes.
CKD9_STRICT_REGEX = r"^585"
CKD10_STRICT_REGEX = r"^N18"
CKD_STRICT_REGEX = r"^(?:585|N18)"


TARGET_TO_REGEX = {
    "DM": DM_REGEX,
    "CVD": CVD_STRICT_REGEX,
    "CKD": CKD_STRICT_REGEX,
}


def normalize_target(target: str) -> str:
    """Normalize user input (dm/cvd/DM/CVD) to canonical key."""
    t = (target or "").strip().upper()
    if t in ("DM", "CVD", "CKD"):
        return t
    raise ValueError(f"Unknown target={target!r}. Use 'dm', 'cvd', or 'ckd'.")


def get_target_regex(target: str) -> str:
    """Return regex string for the chosen target disease."""
    t = normalize_target(target)
    return TARGET_TO_REGEX[t]


