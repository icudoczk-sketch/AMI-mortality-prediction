# AMI 30-day mortality — analysis code

Companion repository for:

> **Prediction of 30-Day In-Hospital Mortality in Patients with Acute Myocardial Infarction:
> A Machine Learning and Feature-Selection Comparison Study Based on the MIMIC-IV Database**
> PLOS ONE, manuscript PONE-D-25-30481

## Where the code is

| Location | Contents |
|---|---|
| `sql/01_arterial_ph.sql` | Arterial blood-gas pH re-extraction (specimen-filtered, earliest value within the first 24 h of ICU admission) |
| `py/02_fix_plr.py` | Platelet-to-lymphocyte ratio recomputed from the absolute lymphocyte count |
| `py/04_figure_spec_check.py` | Checks every figure against the PLOS ONE specifications (300–600 dpi, 2.63–7.5 × 8.75 in, ≤10 MB) |
| Release assets | Data-extraction outputs, derived-variable tables and modelling scripts (Python/R) underlying the analysis |

## Data

MIMIC-IV v2.2 (Johnson et al., *Sci Data* 2023), available from PhysioNet under credentialed access.
The data are not redistributed here; the Data Use Agreement prohibits it. Reproducing the tables requires
PhysioNet credentialing, completion of the CITI "Data or Specimen Only Research" course, and signed
acceptance of the MIMIC-IV DUA.

## Notes on two derived variables

**pH.** Blood-gas pH must be restricted to arterial specimens (`specimen = 'ART'`). A label-based match
across all specimen types mixes urine, pleural and other fluids into the same column and produces values
that are not physiologically possible for arterial blood.

**PLR.** PLR is the platelet count divided by the *absolute* lymphocyte count
(white-cell count × lymphocyte percentage / 100). Dividing by the lymphocyte percentage instead rescales
the ratio by the white-cell count and makes it non-comparable with published values (approximately
150–250 in AMI and ICU cohorts).

## Licence

MIT (see [LICENSE](LICENSE)). The licence covers the code only, not the data.
