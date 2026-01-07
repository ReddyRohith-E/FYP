# Paper Outline and Submission Plan

Title (working): Early ASD Risk Prediction from rs-fMRI in Infants and Toddlers Using Hybrid CNN–LSTM on Dynamic Functional Connectivity

## 1. Abstract (150–250 words)

- Problem, data, approach (sFC, dFC, hybrid), evaluation design (subject-wise), main metrics (AUC, sensitivity), key findings, limitations, and potential clinical relevance.

## 2. Introduction

- Clinical motivation for earlier ASD detection.
- rs-fMRI as a window into functional organization during early neurodevelopment.
- Brief review: sFC, dFC, CNNs on FC matrices, sequence models for dFC, hybrid models; infant/toddler data challenges.
- Contributions:
  - Subject-aware dFC pipeline with CNN+LSTM for infants/toddlers.
  - Rigorous leakage prevention and confound mitigation recommendations.
  - Interpretable connectivity findings mapped to Harvard–Oxford regions.
  - Reproducible code and evaluation splits.

## 3. Methods

### 3.1 Data and Atlas

- Describe cohort and data format (ROI time series CSVs). Reference Harvard–Oxford atlas file: Dataset/Harvard-Oxford Atlas (Label of Brain Regions).csv
- Include subject demographics if available; mention motion metrics if computed.

### 3.2 Preprocessing

- Standardization per subject; optional nuisance regression if applied; windowing (size/stride rationale).
- Train/val/test subject-wise splits using GroupShuffleSplit.

### 3.3 Feature Construction

- sFC: correlation matrices for full time series (static model input).
- dFC: sliding-window correlations producing sequences of FC matrices.

### 3.4 Models

- Static 1D CNN on ROI time series.
- DFC-CNN on FC matrices.
- Hybrid CNN+LSTM: CNN encodes per-window FC; LSTM integrates over time.
- Training details: loss, optimizer, epochs, early stopping or not, class weights.

### 3.5 Evaluation

- Metrics: accuracy, ROC–AUC, sensitivity/specificity, F1; calibration (optional); 95% CI via bootstrapping.
- Leakage controls: strict subject-wise splits; no window leakage.
- Statistical tests: permutation test; DeLong for AUC comparison.

## 4. Experiments

- Data splits: subject counts and class balance per split.
- Baselines and ablations:
  - sFC vs dFC vs hybrid.
  - Window size (e.g., 32/64/96) and stride sensitivity.
  - Subject-wise vs global standardization.
  - Capacity/regularization (dropout, weight decay) sweeps.
- Robustness: shorter scan segments; noise/motion augmentation.
- External/site generalization: leave-site-out if site labels exist.

## 5. Results

- Primary table: mean±CI for accuracy, AUC, sensitivity/specificity for each model.
- Curves: ROC, PR; learning curves.
- Calibration: reliability plots and Brier score (optional).
- Reference preliminary outputs in repo: static_cnn_results.txt, dfc_results.txt, final_comparison_results.txt (to be validated under subject-wise CV).

## 6. Interpretation

- Saliency/attribution on FC edges and ROI importance.
- Summarize network-level effects (default mode, salience, sensorimotor) and link to literature.

## 7. Discussion

- Compare to prior work; highlight pediatric-specific contributions.
- Practical implications for screening; limitations (data size, site effects, motion, generalizability).
- Future work: multimodal integration (DTI/structural), uncertainty quantification, larger cohorts.

## 8. Ethics and Reproducibility

- Statement on clinical vs research use.
- Code/data availability; seeds; detailed configs; split manifests.

## 9. Conclusion

- Concise recap of findings and potential impact.

## 10. References

- Curated during write-up (include infant/toddler rs-fMRI ASD papers, dFC, BrainNetCNN, GNNs, calibration, ComBat harmonization).

---

# Recommended Experiments To Reach Publishable Quality

1. Nested subject-wise CV (or repeated subject-wise splits); report mean±95% CI.
2. Permutation tests (n≥1000) for significance of primary model.
3. Ablations: window size/stride; remove LSTM; remove CNN; standardization variants.
4. Confounds: regress site/age/motion effects or stratify; sensitivity analyses.
5. Robustness: truncated scans (50–75% of time points); noise/motion augmentation.
6. Interpretability: integrated gradients/Grad-CAM; edge-wise importance maps.
7. Calibration: reliability curves, Brier score; threshold tuning for sensitivity.

---

# Target Journals And Submission Strategy

- Frontiers in Neuroscience (Neuroimaging or Brain Imaging Methods): method-focused, open access.
- Scientific Reports: broad scope, solid methods and robustness acceptable.
- IEEE Access: engineering-oriented biomedical applications; fast decisions.
- Children (MDPI) / Brain Sciences (MDPI): neurodevelopment focus (verify scope fit and APCs).
- PLOS ONE: rigorous methodology and data transparency prioritized.

Submission plan:

- Tighten experimental rigor (as above) and recalibrate results.
- Prepare clean figures (FC maps, ROC/PR, calibration) and split manifests.
- Share a reproducible repository with environment lockfile and scripts.
- Draft, internal review, and pre-submission inquiry if appropriate.
