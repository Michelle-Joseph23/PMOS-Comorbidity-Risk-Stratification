# Comorbidity Risk Stratification in Polyendocrine Metabolic Ovarian Syndrome (PMOS) Using Explainable Machine Learning

**Author:** Michelle Ellen Joseph (23BCE1381)

---

## Overview

Polyendocrine Metabolic Ovarian Syndrome (PMOS), referred to in this project as the condition traditionally known as Polycystic Ovary Syndrome (PCOS), extends beyond reproductive health. It is associated with metabolic complications, including type 2 diabetes mellitus (T2DM), obesity, and hypertension, as well as mental health concerns such as depression and anxiety.

Most machine learning approaches to PCOS focus on identifying whether a patient has the syndrome. This project investigates a different problem:

**Among patients already diagnosed with PMOS, can machine learning identify those at higher risk of metabolic and mental health comorbidities, and explain the factors influencing those predictions?**

The project develops and evaluates classical machine learning classifiers, a custom one-dimensional convolutional neural network (1D-CNN), and a quantum-kernel support vector machine (SVM). SHAP and LIME are used to examine model predictions and improve their interpretability.

The long-term objective is to support research into explainable, patient-level comorbidity risk stratification using routinely collected clinical data.

> **Research disclaimer:** This is an experimental machine learning study, not a clinically validated diagnostic or risk prediction system. The comorbidity targets are constructed proxies rather than confirmed clinical diagnoses.

## Objectives

- Develop machine learning models for T2DM-related risk stratification among PMOS-positive patients.
- Investigate depression/anxiety-related risk using a constructed symptom-burden proxy.
- Compare classical classifiers, a stacking ensemble, a 1D-CNN, and a quantum-kernel SVM.
- Apply leakage-aware preprocessing and training-only resampling.
- Interpret model behaviour using global SHAP explanations and patient-level LIME explanations.
- Investigate the impact of removing features used to construct the target labels.
- Evaluate whether quantum-kernel methods offer useful predictive trade-offs against classical baselines.

## Dataset

The study uses a publicly available Kaggle PCOS dataset containing clinical records from hospitals in Kerala, India.

| Property | Description |
|---|---|
| Total records | 541 patients |
| Clinical variables | 44 |
| Study population | 177 PMOS-positive patients |
| Data type | Structured clinical and lifestyle data |
| Geographic coverage | Kerala, India |
| Primary tasks | T2DM-related and depression/anxiety-related risk stratification |

The dataset includes hormonal measurements, anthropometric variables, vital signs, ultrasound findings, and lifestyle-related information.

### Target Variables

The two comorbidity targets are constructed from available dataset variables.

| Prediction task | Operational definition | Positive cases |
|---|---|---:|
| T2DM-related risk | Random blood sugar ≥ 110 mg/dL | 30/177 (16.9%) |
| Depression/anxiety-related risk | Constructed score ≥ 3 out of 5, based on symptom burden, BMI, menstrual-cycle irregularity, hair loss, and lifestyle | 116/177 (65.5%) |

**Important:** These definitions do not establish clinical diagnoses. The dataset does not contain the PHQ-9, GAD-7, HbA1c, or insulin measurements required for the corresponding clinical assessments. In particular, random blood sugar ≥ 110 mg/dL is an operational threshold used in this experiment, not a validated diagnostic threshold for diabetes.

The depression/anxiety target is a researcher-defined proxy and must not be interpreted as a validated mental health screening instrument.

## Methodology

### 1. Data Preprocessing and Feature Engineering

The preprocessing pipeline includes:

- Numeric coercion for clinical variables.
- Median imputation for missing numeric values.
- Engineering of five additional features:
  - Pulse pressure.
  - BMI × waist-to-hip ratio.
  - hCG ratio.
  - Total follicle count.
  - Symptom burden.

Feature definitions, units, and handling of missing or invalid values should be documented alongside the implementation.

### 2. Leakage Control and Class Imbalance

To reduce target leakage:

- Variables directly used to construct each target are excluded from the corresponding feature set.
- SMOTE with `k = 3` is applied only to training folds.
- Evaluation uses the same five-fold stratified cross-validation configuration, with random seed `42`, for model comparison.

The depression/anxiety task is evaluated using two feature configurations:

- **38-feature configuration:** The broader feature set used in the initial experiment.
- **Strict 27-feature configuration:** A reduced feature set intended to remove variables contributing directly to the constructed label.

The strict configuration provides a more conservative assessment of whether the model can predict the proxy target without relying on its defining ingredients. However, removing direct label ingredients does not, by itself, eliminate every possible source of leakage or confounding.

### 3. Predictive Models

#### Classical Machine Learning

Eight classical classifiers are evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine with an RBF kernel
4. Decision Tree
5. Gaussian Naive Bayes
6. Random Forest
7. Gradient Boosting
8. XGBoost

A stacking ensemble is also included to investigate whether combining model predictions improves performance.

#### Deep Learning

A custom one-dimensional convolutional neural network (1D-CNN) is implemented using TensorFlow/Keras. Its architecture includes:

- Two convolutional blocks.
- Batch normalization.
- Global average pooling.
- Dropout.

The CNN is evaluated against the same prediction tasks to assess whether a deep learning approach generalizes better than classical models on this relatively small structured dataset.

#### Quantum Machine Learning

A quantum-kernel SVM is implemented using PennyLane with a four-qubit quantum feature map based on ZZ interactions.

The quantum kernel is evaluated using a noiseless simulator and compared against classical models on the same prediction tasks.

This experiment investigates predictive performance and computational trade-offs. It does **not** establish quantum advantage or demonstrate superiority on quantum hardware.

### 4. Explainability

Two explainability methods are used with the Random Forest model:

- **SHAP (SHapley Additive exPlanations):** Used to examine global feature importance and the contributions of individual features to predictions.
- **LIME (Local Interpretable Model-agnostic Explanations):** Used to explain individual patient-level predictions through local surrogate models.

These explanations describe how the fitted models use the available data. They do not establish causal relationships or clinical risk factors.

## Results

### Cross-Validation Performance

The primary reported metric is the area under the receiver operating characteristic curve (AUROC), evaluated using five-fold stratified cross-validation.

| Model | T2DM-related risk | Depression/anxiety proxy (38 features) | Depression/anxiety proxy (strict 27 features) |
|---|---:|---:|---:|
| Random Forest | **0.603** | **0.800** | **0.608** |
| Quantum-kernel SVM | 0.602 | 0.692 | 0.551 |
| Gradient Boosting | 0.598 | 0.761 | 0.543 |
| XGBoost | 0.579 | 0.787 | 0.526 |
| 1D-CNN | 0.508 | 0.689 | N/A |


### Key Findings

#### 1. Classical Tree Ensembles Were Strong Overall

Random Forest achieved the highest reported AUROC among the listed models for both the T2DM-related task and the depression/anxiety proxy task.

Its performance declined from `0.800` to `0.608` when the depression/anxiety feature set was restricted, indicating that the broader configuration likely benefited substantially from information associated with the construction of the target.

#### 2. The Quantum-Kernel SVM Showed a Different Classification Trade-Off

For the T2DM-related task, the quantum-kernel SVM achieved an AUROC of `0.602`, compared with `0.603` for Random Forest.

The reported quantum-kernel SVM results also showed:

| Metric | Quantum-kernel SVM | Random Forest |
|---|---:|---:|
| Recall | 0.50 | 0.07 |
| F1-score | 0.30 | 0.08 |
| AUROC | 0.602 | 0.603 |

The quantum-kernel SVM identified a larger proportion of positive cases at the evaluated classification threshold, with higher recall and F1-score. However, AUROC values were nearly identical.

These results illustrate the importance of examining threshold-dependent metrics alongside AUROC, particularly when positive cases are uncommon. The reported metrics do not establish that the quantum model is universally superior.

#### 3. The Depression/Anxiety Proxy Was Sensitive to Feature Selection

The Random Forest AUROC fell from `0.800` with the broader 38-feature configuration to `0.608` with the strict 27-feature configuration.

This substantial decline suggests that the broader model may have learned patterns associated with the variables used to define the proxy target rather than independently predicting a clinically established mental health outcome.

**The strict-feature result is therefore the more conservative result to emphasize**, while recognizing that the proxy label itself remains unvalidated.

#### 4. The 1D-CNN Did Not Demonstrate Strong Generalization

The 1D-CNN achieved AUROCs of `0.508` for the T2DM-related task and `0.689` for the broader depression/anxiety proxy task.

The experiments indicated that the network memorized training data without achieving consistently strong generalization. Its additional architectural complexity did not translate into better predictive performance than the strongest classical models.

#### 5. T2DM-Related Prediction Remained Weak

All listed models achieved AUROCs close to `0.50–0.60` on the T2DM-related task. Their results indicate limited discriminative performance using the available features and constructed target.

The current model is **not suitable for clinical deployment**. More informative clinical measurements, improved outcome definitions, and independent validation are needed before meaningful clinical conclusions can be drawn.

#### 6. PMOS Diagnosis Served as a Separate Sanity Check

As a separate experiment, the same general pipeline achieved approximately `91%` accuracy and `0.96` AUROC for PMOS diagnosis using all 541 patients.

This result concerns a different prediction task and population. It should not be interpreted as evidence that the comorbidity models achieve comparable performance.

The diagnostic experiment also requires its own leakage assessment and independent validation before its results can be interpreted clinically.

## Limitations

- **Small, geographically limited cohort:** The dataset contains 541 records from hospitals in one Indian state, with only 177 PMOS-positive patients used for the primary tasks. It is not representative of a population-level cohort.
- **Constructed outcome labels:** Neither target is equivalent to a confirmed clinical diagnosis. The mental health proxy, in particular, requires independent validation.
- **Limited T2DM-positive cases:** Only 30 patients meet the experimental T2DM-related threshold, making performance estimates sensitive to individual observations and fold composition.
- **Potential residual leakage:** Excluding direct label-defining variables reduces one important source of leakage but cannot guarantee that all indirect leakage has been removed.
- **Feature selection and validation:** Feature selection was not nested within cross-validation, and the main results use a single five-fold cross-validation configuration. This can introduce selection bias and uncertainty into performance estimates.
- **Limited uncertainty analysis:** The reported table does not include confidence intervals or a comprehensive analysis of fold-to-fold variability.
- **Simulated quantum computation:** The quantum-kernel experiment uses four qubits on a noiseless simulator. It does not demonstrate a computational or predictive advantage on real quantum hardware.
- **Deep learning generalization:** The small dataset limits the ability to train and evaluate a comparatively complex neural network reliably.
- **No external validation:** The models have not been validated on an independent hospital, geographic region, or prospective patient cohort.
- **Clinical interpretability is not clinical validity:** SHAP and LIME explain model behaviour but do not establish causal mechanisms, clinical significance, or patient benefit.

## Future Work

1. **Use clinically measured outcomes:** Replace proxy targets with appropriate clinical measurements, including HbA1c and fasting glucose for glycemic assessment, relevant insulin measurements, and validated PHQ-9 and GAD-7 assessments for depression and anxiety.
2. **Expand the dataset:** Evaluate larger, multicentre cohorts with greater demographic and clinical diversity.
3. **Strengthen validation:** Use repeated, nested cross-validation, confidence intervals, calibration curves, and threshold-specific metrics.
4. **Conduct external validation:** Test the final models on independent datasets and assess their generalizability across clinical settings.
5. **Investigate longitudinal prediction:** Follow patients over time to study transitions into prediabetes and other metabolic or mental health outcomes.
6. **Improve clinical interpretability:** Review feature definitions and explanations with qualified clinicians and assess whether the explanations are stable and clinically meaningful.
7. **Extend quantum experiments:** Investigate additional qubit configurations, noise models, computational costs, and real quantum hardware where feasible.
8. **Explore multimodal learning:** Investigate combining structured clinical data with ultrasound images, subject to sufficient data availability and rigorous validation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Classical machine learning | scikit-learn |
| Class imbalance | imbalanced-learn (SMOTE) |
| Gradient boosting | XGBoost |
| Deep learning | TensorFlow / Keras |
| Quantum machine learning | PennyLane |
| Global explainability | SHAP |
| Local explainability | LIME |
| Visualization | Matplotlib, Seaborn |

## Repository Structure

The following is a suggested structure. Update it to match the files actually present in the repository.

```text
.
├── data/
│   └── README.md                 # Dataset access and preparation instructions
├── notebooks/
│   ├── classical_models.ipynb    # Classical classifiers and ensemble
│   ├── cnn_model.ipynb           # 1D-CNN experiments
│   ├── quantum_kernel.ipynb      # Quantum-kernel SVM
│   └── explainability.ipynb      # SHAP and LIME analysis
├── figures/
│   ├── roc_curves/
│   ├── shap/
│   ├── lime/
│   └── confusion_matrices/
├── requirements.txt              # Python dependencies
└── README.md
```


## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the selected package releases.
- Jupyter Notebook or JupyterLab.
- Access to the source dataset on Kaggle.

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the relevant notebook under `notebooks/` and follow its dataset preparation and execution instructions.

**Reproducibility note:** The commands above describe the intended workflow. A complete reproducible release should specify tested package versions, provide the actual dependency file, document dataset preparation, and record all random seeds and cross-validation settings.

## Dataset Source

**Source:** Public Kaggle dataset titled *Polycystic Ovary Syndrome (PCOS)*, reported to contain 541 patient records from hospitals in Kerala, India.

Find the dataset on [Kaggle]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos).

License: CC BY-NC-SA 4.0

## Ethical and Clinical Considerations

This project is an exploratory research study using a small, geographically limited dataset and researcher-defined proxy targets. Its outputs must not be used to diagnose diabetes, depression, anxiety, PCOS/PMOS, or any other condition, or to make treatment decisions.

Clinical translation would require validated outcomes, representative data, rigorous independent evaluation, calibration, assessment of subgroup performance, privacy safeguards, and review by qualified healthcare professionals.

## Author

**Michelle Ellen Joseph**  
Computer Science and Engineering  
Vellore Institute of Technology, Chennai

---

*This repository documents an experimental investigation into explainable machine learning and quantum-kernel methods for comorbidity risk stratification. Reported results reflect the specific dataset, feature definitions, and evaluation setup described above and should not be interpreted as evidence of clinical effectiveness or quantum advantage.*
