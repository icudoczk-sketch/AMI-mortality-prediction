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
| `py/03_run_analysis.py` | Full nested pipeline: Tables 1–6 and the artefacts behind Figs 2–6, S1 and S2 |
| `py/03_make_figures.py` | Generates Figs 2–6, S1 and S2 from those artefacts |
| `py/03_sensitivity.py` | Split-stability and collinearity sensitivity analyses |
| `py/04_figure_spec_check.py` | Checks every figure against the PLOS ONE specifications (300–600 dpi, 2.63–7.5 × 8.75 in, ≤10 MB) |
| Release assets | Data-extraction outputs, derived-variable tables and modelling scripts (Python/R) underlying the analysis |

## Data

MIMIC-IV v2.2 (Johnson et al., *Sci Data* 2023), available from PhysioNet under credentialed access.
The data are not redistributed here; the Data Use Agreement prohibits it. Reproducing the tables requires
PhysioNet credentialing, completion of the CITI "Data or Specimen Only Research" course, and signed
acceptance of the MIMIC-IV DUA.

## Notes on two derived variables

**pH.** The pH field must not be built by matching on the label alone. In MIMIC-IV two laboratory items
carry the label "pH": itemid 50820 (blood gas; median 7.37, IQR 7.31-7.42) and itemid 51491 (urine;
median 6.00, IQR 5.50-6.00). Merging them produces values between 5.00 and 6.80 that are incompatible
with arterial blood. `py/06_export_arterial_ph.py` re-extracts the blood-gas item only; a SQL variant
that restricts to arterial specimens is in `sql/01_arterial_ph.sql`.

**PLR.** PLR is the platelet count divided by the *absolute* lymphocyte count
(white-cell count × lymphocyte percentage / 100). Dividing by the lymphocyte percentage instead rescales
the ratio by the white-cell count and makes it non-comparable with published values (approximately
150–250 in AMI and ICU cohorts).

## Licence

MIT (see [LICENSE](LICENSE)). The licence covers the code only, not the data.
