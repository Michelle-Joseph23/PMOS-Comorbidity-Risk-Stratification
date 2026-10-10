# PMOS-Comorbidity-Risk-Stratification

Comorbidity Risk Stratification in PMOS Using Explainable ML

Healthcare Data Analytics (BCSE335L), Winter Semester 2026-27, VIT Chennai
Author: Michelle Ellen Joseph (23BCE1381)

Overview

Polyendocrine Metabolic Ovarian Syndrome (PMOS, earlier called PCOS) goes far beyond the ovaries. Many patients later develop type 2 diabetes, obesity, hypertension, and depression or anxiety. Most machine learning work on PMOS only asks whether a patient has the syndrome. This project asks a different question: among patients who already have PMOS, who is at higher risk of a metabolic or mental health comorbidity, and why?

The goal is a tool that ranks PMOS patients by comorbidity risk using routine clinical records, with explanations a doctor can check.

Dataset

Public Kaggle PCOS dataset from Kerala hospitals: 541 patients, 44 clinical variables (hormones, anthropometry, vitals, ultrasound, lifestyle). Only the 177 PMOS-positive patients are used for the comorbidity tasks.

Target	Definition	At risk
T2DM risk	Random blood sugar ≥ 110 mg/dL	30 / 177 (16.9%)
Depression/anxiety risk (proxy)	Score ≥ 3 of 5 from symptom burden, BMI, cycle irregularity, hair loss, lifestyle	116 / 177 (65.5%)

The dataset has no PHQ-9, GAD-7, HbA1c, or insulin values, so both targets are proxies, not clinical diagnoses.

Pipeline
Preprocessing: median imputation, numeric coercion, five engineered features (pulse pressure, BMI × waist-hip ratio, hCG ratio, total follicles, symptom burden).
Leakage control: the variable behind each label is removed from its feature set. SMOTE (k = 3) is applied only inside training folds.
Models, all evaluated on the same 5-fold stratified CV (seed 42):
Contribution 1: eight classical classifiers (Logistic Regression, KNN, SVM-RBF, Decision Tree, Naive Bayes, Random Forest, Gradient Boosting, XGBoost) plus a stacking ensemble.
Contribution 2: a small custom 1D-CNN (two Conv blocks, batch norm, global average pooling, dropout).
Contribution 3: a four-qubit quantum-kernel SVM (ZZ feature map, PennyLane, noiseless simulator).
Explainability: SHAP (global) and LIME (per patient) on the Random Forest.
Key Results (5-fold CV, AUROC)
Model	T2DM risk	Depression/anxiety (38 features)	Depression/anxiety (strict, 27 features)
Random Forest	0.603	0.800	0.608
Quantum-kernel SVM	0.602	0.692	0.551
Gradient Boosting	0.598	0.761	0.543
XGBoost	0.579	0.787	0.526
1D-CNN	0.508	0.689	n/a

Main takeaways

Tree ensembles were the strongest overall.
The quantum kernel matched Random Forest on T2DM with much higher recall (0.50 vs 0.07) and F1 (0.30 vs 0.08).
The 1D-CNN memorised the training data and did not generalise.
SHAP showed the depression label could be partly recovered from its own ingredients. After removing them (strict feature set), AUROC fell from 0.80 to about 0.61. The honest depression result is the lower one.
T2DM risk is close to chance with these routine features, so it is not clinically usable as it stands.
As a sanity check, the same pipeline on PMOS diagnosis (all 541 patients) reaches about 91% accuracy and AUROC 0.96.
Limitations
Small, single-region cohort (not truly population-level).
Both targets are proxy labels.
Feature selection was not nested, and the main tables use one 5-fold split.
The quantum model is simulated with four qubits, so nothing here says anything about hardware advantage.
Held-out single-split curves are noisy (only 8 at-risk patients in the T2DM test set), so cross-validation is the reported result.
Future Work
Replace proxy labels with measured outcomes (HbA1c, fasting glucose and insulin, PHQ-9, GAD-7).
Use a larger multi-centre dataset with external validation and clinician review of explanations.
Repeated, nested cross-validation with confidence intervals and calibration curves.
Longitudinal follow-up to model transitions into pre-diabetes.
More qubits, noise models, and real hardware; fusion with ultrasound images.
Tech Stack

Python, scikit-learn, imbalanced-learn (SMOTE), XGBoost, TensorFlow/Keras (1D-CNN), PennyLane (quantum kernel), SHAP, LIME.

Repository Structure

Adjust this to match your repo.

├── data/            # Kaggle PCOS CSV
├── notebooks/       # classical models, 1D-CNN, quantum-kernel SVM
├── figures/         # ROC curves, SHAP, LIME, confusion matrices
└── README.md
Dataset Source

Kaggle: Polycystic Ovary Syndrome (PCOS) dataset (541 patients, Kerala hospitals).
