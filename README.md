<div align="center">

# Kairo Health

### OCR Noise Effects on Medical Record Information Extraction

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-5.x-4285F4?style=flat&logo=google&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Sonnet-D4A373?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

**University of Maryland, College of Information | INST664: Transforming Unstructured Content with AI | Final Project**

[Headline Findings](#headline-findings) · [Statistical Testing](#statistical-testing) · [Deployment Guide](#deployment-guide) · [Pipeline Stages](#pipeline-stages) · [Data and Sources](#data-and-sources)

---

</div>

## Project Overview

Kairo Health started from a question I have been carrying since cofounding Frontground, a startup focused on personalized mobile electronic medical records for low resource clinics. Mobile capture and cloud storage solve part of the problem, but they leave a harder upstream issue untouched: once you have a photo of a paper medical record, how do you actually use it? A blurry JPEG of a triage form is not the same as structured patient data. This project is a controlled study of that conversion step.

The pipeline generates synthetic pediatric triage forms modeled on the Médecins Sans Frontières Aweil Pediatric Triage form. Each form is rendered at five OCR noise tiers: clean, moderate, heavy, severe, and extreme. Tesseract OCR is then run on every image, and two extraction methods are compared on the OCR text: a rule based regex baseline and a zero shot LLM extractor using Claude Sonnet 4.5.

The evaluation reports precision, recall, and F1 at the field level and across noise tiers. It also adds bootstrap confidence intervals, McNemar significance tests, and per field error categorization for missing, substitution, hallucination, and layout artifact failures.

> **Core question:** As OCR input quality degrades across a controlled five tier noise spectrum, does LLM based field extraction stay more reliable than a rule based baseline, and where does each method break?

The headline finding is that the methods are statistically indistinguishable on clean and moderate inputs. The LLM shows a significant and growing advantage through heavy and severe noise. At extreme noise both methods fail because OCR quality makes the document almost unreadable, and the LLM's advantage there is not statistically significant. The LLM also shows strong grounding behavior overall: across 2,100 field extraction attempts, only 5 hallucinations were observed, all at the two most degraded tiers.

---

## Headline Findings

| Noise | Sorted CER | Rules F1 | LLM F1 | Rules Recall | LLM Recall |
|---|---:|---:|---:|---:|---:|
| Clean | 0.32 | 0.93 | 0.91 | 0.90 | 0.91 |
| Moderate | 0.32 | 0.92 | 0.91 | 0.89 | 0.91 |
| Heavy | 0.48 | 0.86 | 0.91 | 0.77 | 0.90 |
| Severe | 0.80 | 0.58 | 0.67 | 0.42 | 0.54 |
| Extreme | 0.94 | 0.00 | 0.02 | 0.00 | 0.01 |

The recall column tells the clearest version of the story. From clean to heavy noise, LLM recall drops by only 1 point, from 0.91 to 0.90, while rules recall drops by 13 points, from 0.90 to 0.77. At severe noise, the gap widens: rules recall falls to 0.42 while LLM recall holds at 0.54. At extreme noise both methods fail. The ground truth has a value in every field, so the LLM returning null for 410 of 420 fields is 410 misses rather than 410 correct refusals, which is exactly why its recall there is 0.01. What the behavior does show is that the model goes quiet instead of inventing values from medical priors. The rules method returned a value on only 2 of 420 fields and both were wrong.

> **Note on CER:** Sorted CER measures character recognition quality after token sorting both the OCR output and reference text. This removes reading order penalties caused by Tesseract's column traversal. Raw CER is also computed and stored in `data/ocr_results.json`, but sorted CER is the cleaner metric for this analysis.

---

## Statistical Testing

Bootstrap confidence intervals with 1,000 paired resamples and McNemar exact tests confirm that the visual pattern in the headline table is statistically meaningful.

| Noise | LLM minus Rules F1 | 95% CI | Significant |
|---|---:|---|---|
| Clean | -0.013 | [-0.038, +0.010] | No |
| Moderate | -0.008 | [-0.029, +0.014] | No |
| Heavy | +0.042 | [+0.019, +0.066] | **Yes** |
| Severe | +0.093 | [+0.061, +0.123] | **Yes** |
| Extreme | +0.024 | [+0.000, +0.059] | No |

The methods are statistically indistinguishable on clean and moderate inputs. The LLM advantage becomes significant at heavy and severe noise, with the largest gap at severe noise. At extreme noise, both methods are essentially broken, so the advantage disappears.

McNemar field level testing at heavy noise produced p = 1.6e-15, with 55 fields correct under LLM only versus 1 field correct under rules only among the 56 disagreements. In other words, when these two methods see the same heavy noise input and disagree, the LLM is right 55 times out of 56. The severe tier shows the same pattern at 54 versus 4 among 58 disagreements, p = 3.2e-12.

See `results/bootstrap_ci.csv`, `results/bootstrap_diff.csv`, and `results/mcnemar.csv`.

---

## Deployment Guide

Error analysis at heavy, severe, and extreme noise categorizes each failure as missing, substitution, hallucination, or layout artifact. The table below reports the heavy tier on its own, because that is the tier where the two methods separate while both are still usable. Note that `results/deployment_guide.csv` currently pools all three degraded tiers, so its totals are larger than the heavy only counts shown here.

| Category | Meaning |
|---|---|
| Missing | Method returned nothing even though gold had a value |
| Substitution | Method returned a wrong value that appears in the OCR text |
| Hallucination | Method returned a wrong value that does not appear in the OCR text |
| Layout artifact | Method returned nothing and the field label was absent from OCR text |

### Field Level Recommendation at Heavy Noise

| Field | Rules Errors | LLM Errors | Recommended |
|---|---:|---:|---|
| date | 0 | 0 | either |
| name | 0 | 0 | either |
| sat | 0 | 0 | either |
| muac | 0 | 0 | either |
| hr | 1 | 1 | either |
| temperature | 3 | 3 | either |
| age | 1 | 0 | llm |
| rr | 1 | 0 | llm |
| time | 1 | 0 | llm |
| presenting_complaint | 1 | 0 | llm |
| weight | 4 | 3 | llm |
| ampm | 25 | 17 | llm |
| sex | 30 | 13 | llm |
| triage_color | 29 | 5 | llm |

The LLM matches or beats the rules baseline on every field at heavy noise. The practical recommendation is to use regex on the six fields where the two methods tie, because those are deterministic, auditable, and work with no network connection, and to route the other eight through the LLM.

Routing this way scores F1 0.906 at heavy noise, identical to sending every field to the model. It is worth being precise about what that does and does not save. The API call is per document rather than per field, so one call already returns all fourteen values and moving six of them to regex saves output tokens and little else. The gain is that six fields become deterministic and keep working offline, not that the pipeline gets meaningfully cheaper. The split is also tuned at heavy noise and should not be applied to clean input, where the pure regex baseline scores higher than either alternative at F1 0.927.

**Safety caveat:** The LLM classified 25 of 30 forms correctly at heavy noise. Of its 5 errors on `triage_color`, 3 were over triage, meaning GREEN was predicted as YELLOW, which is the safer direction. The other 2 were under triage, meaning RED was predicted as YELLOW. The rules method returned nothing on this field for 29 of 30 forms, which silently propagates as null triage data downstream.

Severe noise is worse and the pattern is different. The LLM classified 12 of 30 correctly, left 13 forms unclassified including 4 marked RED by the clinician, and produced one RED to GREEN error. Both failure modes argue against autonomous triage classification at any noise level. See `results/triage_safety_matrix.png` in `docs/assets`.

### Hallucination Profile Across All Tiers

| Tier | LLM Hallucinations | Total Field Attempts |
|---|---:|---:|
| Clean | 0 | 420 |
| Moderate | 0 | 420 |
| Heavy | 0 | 420 |
| Severe | 2 | 420 |
| Extreme | 3 | 420 |
| **Total** | **5** | **2,100** |

There were zero hallucinations below severe OCR quality. At extreme noise the LLM returns null for almost every field rather than guessing from medical priors. Those nulls are scored as misses, not as correct abstentions, because every field in the ground truth has a value.

See `results/error_categories.csv` and `results/deployment_guide.csv`.

---

## Data and Sources

<details>
<summary><strong>Source Form</strong></summary>

<br>

The synthetic dataset is modeled on the MSF Aweil Pediatric Triage form, a clinical document used by Médecins Sans Frontières clinicians for pediatric triage in low resource settings. I selected this form because clinicians I interviewed during Frontground customer discovery, specifically at the MSF clinic in Monrovia, used a near identical variant.

| Property | Description |
|---|---|
| Source | MSF Medical Guidelines |
| Form name | MSF Aweil Pediatric Triage |
| Use context | Pediatric outpatient triage in low resource clinics |
| Structural regions | Header and patient identifiers, color coded clinical assessment grid, vitals table |

</details>

<details>
<summary><strong>Synthetic Data Generation</strong></summary>

<br>

Each form is filled with medically plausible patient data: ages sampled from a pediatric distribution, vitals in realistic ranges calibrated by age band, plausible presenting complaints, and triage classifications consistent with the sampled vital signs.

| Field Type | Examples | Count |
|---|---|---:|
| Header and identifiers | date, time, ampm, name, age, sex | 6 |
| Free text | presenting_complaint | 1 |
| Checkbox classification | triage_color | 1 |
| Numeric vitals | rr, hr, sat, temperature, weight, muac | 6 |
| Total evaluation fields | | **14** |

</details>

<details>
<summary><strong>OCR Noise Tiers</strong></summary>

<br>

Each PDF is rendered at five noise tiers through parameterized image degradation. Sorted CER measures character recognition quality independent of reading order differences between Tesseract output and the reference text.

| Tier | Rotation | Blur | Contrast | Smudges | Tint | Sorted CER |
|---|---:|---:|---:|---:|---|---:|
| Clean | 0deg | 0px | 1.00 | 0 | none | 0.32 |
| Moderate | 0.5deg | 0.5px | 0.85 | 0 | none | 0.32 |
| Heavy | 1.5deg | 1.0px | 0.70 | 0 | yellow +10 | 0.48 |
| Severe | 3.0deg | 1.5px | 0.55 | 3 | yellow +20 | 0.80 |
| Extreme | 5.0deg | 2.5px | 0.40 | 8 | yellow +30 | 0.94 |

</details>

<details>
<summary><strong>Techniques</strong></summary>

<br>

| Technique | Library | Purpose |
|---|---|---|
| PDF generation | reportlab | Synthesize triage form PDFs |
| PDF to image | pdf2image and poppler | Convert PDFs for OCR pipeline |
| Image degradation | PIL and numpy | Apply controlled noise per tier |
| OCR | pytesseract and Tesseract 5.x | Extract raw text from noisy images |
| OCR error metrics | jiwer | Raw CER, sorted CER, and word error rate |
| Rule based extraction | re | Baseline label anchored extraction |
| LLM extraction | Anthropic API and Claude Sonnet 4.5 | Zero shot JSON extraction |
| Statistical testing | scipy and numpy | Bootstrap confidence intervals and McNemar exact test |
| Evaluation | pandas, matplotlib, seaborn | Precision, recall, F1, heatmaps, and bar charts |

</details>

---

## Setup Instructions

### System Dependencies

```bash
# macOS
brew install tesseract poppler

# Ubuntu or Debian
sudo apt-get install tesseract-ocr poppler-utils
```

### Python Environment

```bash
git clone https://github.com/kennethyeaher/kairoHealth.git
cd kairoHealth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API Key

```bash
cp .env.example .env
# edit .env and add your Anthropic key
```

Cost for the full 30 form, 5 tier pipeline is under $2 at current Claude Sonnet 4.5 pricing. You only pay it if you change the prompt or the model, because the committed response cache already covers every document in the published run.

---

## Running the Project

```bash
python -m src.generate_forms    # PDFs and ground truth
python -m src.degrade_images    # five noise tiers per PDF
python -m src.run_ocr           # Tesseract plus raw and sorted CER
python -m src.extract_rules     # regex predictions
python -m src.extract_llm       # LLM predictions, served from data/llm_cache
python -m src.evaluate          # tables and figures
python -m src.significance      # bootstrap CIs and McNemar tests
python -m src.error_analysis    # failure categorization and deployment guide
python -m src.make_figures      # case study figures in docs/assets
```

`src.extract_llm` reads `data/llm_cache` before calling the API, so a fresh clone reproduces the published predictions with no key and no spend. Pass `--refresh` to call the API and overwrite the cache, or `--no-cache` to bypass it entirely. The cold run that produced the committed cache took 5 minutes.

---

## Code Package Structure

| Type | Path | Description |
|---|---|---|
| **Folder** | **`src/`** | **Pipeline modules** |
| File | `generate_forms.py` | Synthesizes MSF style triage PDFs |
| File | `degrade_images.py` | Renders PDFs at 5 OCR noise tiers |
| File | `run_ocr.py` | Runs Tesseract OCR with raw and sorted CER |
| File | `extract_rules.py` | Regex baseline extractor |
| File | `extract_llm.py` | Zero shot LLM extractor using Claude API |
| File | `evaluate.py` | Precision, recall, F1, tables, and figures |
| File | `significance.py` | Bootstrap confidence intervals and McNemar exact tests |
| File | `error_analysis.py` | Failure categorization and deployment guide |
| File | `make_figures.py` | Builds the case study figures from `results/` |
| **Folder** | **`data/`** | **Pipeline data artifacts** |
| File | `ground_truth.json` | Field values for all 30 forms |
| File | `ocr_results.json` | Raw Tesseract output with CER per tier |
| Subfolder | `pdfs/` | Generated forms, gitignored and regeneratable |
| Subfolder | `images/` | Noisy renders, gitignored and regeneratable |
| Subfolder | `llm_cache/` | One raw API response per document, committed for replay |
| **Folder** | **`results/`** | **Tables and figures** |
| File | `f1_comparison.csv` | Headline precision, recall, and F1 table across all 5 tiers |
| File | `per_field_f1.csv` | Per method, per noise, per field F1 |
| File | `ocr_calibration.csv` | Mean raw CER, sorted CER, and WER per tier |
| File | `predictions_rules.json` | Regex predictions for all 150 documents |
| File | `predictions_llm.json` | LLM predictions for all 150 documents |
| File | `bootstrap_ci.csv` | F1 with 95% CI per method per noise tier |
| File | `bootstrap_diff.csv` | Paired F1 difference with 95% CI per tier |
| File | `mcnemar.csv` | McNemar exact p value per noise tier |
| File | `error_categories.csv` | Failure counts by method, noise, field, and category |
| File | `deployment_guide.csv` | Recommended method per field at heavy noise |
| File | `f1_by_noise.png` | Headline F1 bar chart |
| File | `recall_by_noise.png` | Recall bar chart |
| File | `rules_field_heatmap.png` | Per field F1 heatmap for regex |
| File | `llm_field_heatmap.png` | Per field F1 heatmap for LLM |
| **Folder** | **`docs/assets/`** | **Figures for the written case study** |
| File | `recall_by_noise.png` | Recall per method per tier with significance marked |
| File | `triage_safety_matrix.png` | Triage confusion at heavy and severe noise |
| File | `field_errors_heavy.png` | Per field errors at heavy noise, both methods |
| File | `noise_ladder.png` | One form header rendered at all five tiers |
| File | `pipeline.svg` | Pipeline diagram |
| File | `config.py` | Paths, seeds, field schema, and noise configs |
| File | `requirements.txt` | Pinned dependencies |
| File | `.env.example` | Template for API key configuration |

---

## Pipeline Stages

```mermaid
flowchart TD
    subgraph GEN ["Data Generation"]
        A([Generate Forms]) --> B([Degrade Images])
    end

    subgraph OCR ["OCR"]
        C([Run Tesseract])
    end

    subgraph EXTRACT ["Extraction"]
        D([Regex Baseline])
        E([LLM Zero Shot])
    end

    subgraph EVAL ["Evaluation"]
        F([Score Precision Recall F1])
        G([Bootstrap and McNemar])
        H([Error Analysis])
        I([Generate Figures])
    end

    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I

    classDef gen fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef ocr fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef rules fill:#ede9fe,stroke:#7c3aed,color:#3b0764
    classDef llm fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef eval fill:#d1fae5,stroke:#059669,color:#064e3b

    class A,B gen
    class C ocr
    class D rules
    class E llm
    class F,G,H,I eval
```

---

<details>
<summary><strong>1. Generate Forms</strong></summary>

<br>

Synthesizes 30 pediatric triage form PDFs using reportlab. Each form is filled with medically plausible patient data sampled with a fixed random seed. The 14 evaluation fields per form are written to `ground_truth.json`.

</details>

<details>
<summary><strong>2. Degrade Images</strong></summary>

<br>

Renders each PDF at 200 DPI and applies one of five degradation pipelines parameterized in `NOISE_CONFIGS`. The output is 150 PNG images, made from 30 forms across 5 tiers. Smudge overlays are introduced at severe and extreme tiers to occlude characters entirely, pushing OCR from noisy text into missing text.

</details>

<details>
<summary><strong>3. Run OCR</strong></summary>

<br>

Runs Tesseract 5.x on each of the 150 images with light preprocessing: grayscale conversion plus 1.2x contrast. Computes raw CER, token sorted CER, and WER per document. Sorted CER is the primary calibration metric because it removes reading order penalties from Tesseract's column traversal.

</details>

<details>
<summary><strong>4. Rule Based Extraction</strong></summary>

<br>

Applies 14 regex patterns to the OCR text, one per evaluation field. Patterns are anchored on field labels and tuned on clean OCR output only. They are applied unchanged at all five noise tiers to preserve the realistic deployment story.

</details>

<details>
<summary><strong>5. LLM Extraction</strong></summary>

<br>

Sends each OCR text to the Anthropic API with a zero shot prompt requesting JSON output matching the field schema. The pipeline uses Claude Sonnet 4.5. The prompt does not ask the model to correct OCR errors. Hallucinated keys outside the schema are dropped during postprocessing and logged in the error analysis.

</details>

<details>
<summary><strong>6. Evaluate</strong></summary>

<br>

Computes true positives, false positives, and false negatives at the method by noise by field level. Wrong predictions count as both FP and FN because emitting a wrong value in a clinical context is more dangerous than emitting nothing. Aggregates produce headline precision, recall, and F1 per method per tier, plus per field heatmaps.

</details>

<details>
<summary><strong>7. Significance Testing</strong></summary>

<br>

Paired bootstrap with 1,000 resamples and seed 42 computes 95% confidence intervals on the LLM minus Rules F1 difference at each noise tier. A field level McNemar exact test counts off diagonal disagreements and reports an exact two sided binomial p value. Both tests run on the same scored document objects so the scoring convention is identical to `evaluate.py`.

</details>

<details>
<summary><strong>8. Error Analysis</strong></summary>

<br>

Categorizes every wrong prediction at heavy, severe, and extreme noise into missing, substitution, hallucination, or layout artifact. The deployment guide aggregates total errors per method per field and recommends the lower error method. Hallucination detection checks whether the predicted value appears anywhere in the OCR text for that document.

</details>

---

## Reproducibility

All randomness is seeded through `RANDOM_SEED = 42` in `config.py`.

| Stage | Reruns Identical | Notes |
|---|---|---|
| Form generation | Yes | Fully deterministic |
| Image degradation | Yes | Seeded random rotation and smudge placement |
| OCR | Yes | Tesseract is deterministic for fixed input |
| Regex extraction | Yes | Pure pattern matching |
| LLM extraction | Yes | Temperature is pinned to 0 and every raw response is cached in `data/llm_cache` |
| Evaluation | Yes | Pure aggregation |
| Significance testing | Yes | Seeded bootstrap generator |
| Error analysis | Yes | Deterministic categorization |

---

## Limitations

This is a synthetic data study on printed forms. Results should be read as an upper bound on extraction quality achievable with clean printed templates. Field performance on real handwritten triage records would almost certainly be lower for both methods.

The two under triage errors at heavy noise, where RED was predicted as YELLOW by the LLM on `triage_color`, are a real safety concern and the clearest argument for mandatory human review of triage classifications. Severe noise adds a RED to GREEN error and four RED forms left unclassified. The rules method is not the safer alternative: its 29 missing values on the same field at heavy noise propagate silently as default routing. Both failure modes argue against autonomous triage decisions from either method.

At extreme noise, where sorted CER is 0.94, both methods fail. The 5 LLM hallucinations observed in the full study are concentrated here. The near zero hallucination rate below extreme noise is a property of this specific prompt and noise range, not a guaranteed property of LLM extractors in general.

Two measurement caveats apply to the noise ladder itself. Clean and moderate score 0.317 and 0.316 sorted CER, which is close enough to treat as one condition, so the study reports five tiers but delivers four distinct ones. And a clean 200 DPI render of machine printed text should OCR near 0.02, not 0.32. The inflation comes from `reconstruct_source_text` in `run_ocr.py`, which compares OCR output against a hand typed reconstruction of the form labels, so every transcription mismatch is charged to Tesseract. The ordering across tiers is meaningful; the absolute values are not.

The LLM stage is reproducible in the sense that matters for this repository: temperature is pinned to 0 and every raw response is committed under `data/llm_cache`, so anyone can regenerate every table and figure here with no API key and no spend. What that does not guarantee is that a fresh call to the API would return the same text a year from now. Model versions change on the provider side, and the cache preserves the answers this study was built on rather than promising the model will answer that way again.

---

## Next Steps and Future Considerations

| Enhancement | Description | Impact |
|---|---|---|
| Real form pilot | Apply the pipeline to a small set of real deidentified triage forms | Test whether the synthetic to real gap matches expectations |
| Few shot LLM variant | Add 2 in context examples to the LLM prompt | Quantify the few shot improvement over zero shot |
| Handwriting OCR | Swap Tesseract for a handwriting capable OCR model such as TrOCR or Google Document AI | Address the largest gap between synthetic and real conditions |
| Vitals cross validation | Flag triage color predictions that contradict extracted vitals | Catch under triage errors without human review of every form |
| Reference text from source | Derive the CER reference from the same draw calls that render the form | Make the OCR quality metric trustworthy in absolute terms |
| Per tier deployment guide | Parameterize the tier in `error_analysis.py` | Stop pooling three tiers into a table labeled heavy noise |

---

<div>

## Author

**Kenneth Yeaher**  
Master of Information Management, Class of 2027  
University of Maryland, College Park  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kenneth_Yeaher-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kennethyeaher/)

`Healthcare NLP` · `Information Extraction` · `OCR` · `LLM Evaluation`

</div>