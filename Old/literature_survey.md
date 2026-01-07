# Literature Survey: Early ASD Detection from fMRI

Problem focus: Developing predictive models for early ASD detection in infants and toddlers using resting-state fMRI (rs-fMRI) time series and derived connectivity features.

## Scope And Motivation

- ASD diagnosis is typically behavioral and occurs after 24–36 months; neuroimaging biomarkers may support earlier risk stratification.
- rs-fMRI characterizes intrinsic functional organization; developmental trajectories in early life are rapid, making age a key confound.
- Practical constraints in infants/toddlers include motion, sleep/sedation, short scans, and small datasets; analysis must be robust and sample-efficient.

## Datasets And Atlases

- Public repositories predominantly feature children/adolescents/adults (e.g., ABIDE I/II); true infant/toddler fMRI cohorts are smaller (e.g., Baby Connectome Project; Developing Human Connectome Project for neonates). Many studies therefore face domain shift when adapting adult-trained models to infants.
- Brain parcellations such as Harvard–Oxford, AAL, Schaefer, and Desikan–Killiany are widely used. Atlas choice and parcel count affect FC feature stability, especially with short scans common in pediatrics.

## Preprocessing And Confound Control

- Standard rs-fMRI steps: motion correction, slice timing, spatial normalization, nuisance regression (WM/CSF/motion), temporal filtering, and scrubbing/censoring. Infant alignment typically requires age-appropriate templates.
- Confounds: age in months, sex, site/scanner, head motion metrics (FD), and acquisition length. Statistical controls and harmonization (e.g., ComBat for multi-site) are recommended.

## Feature Families

- Static Functional Connectivity (sFC): Pearson correlation between region time series; features as upper-triangular vectors or full matrices.
- Dynamic Functional Connectivity (dFC): Sliding-window correlations, time–varying FC states (e.g., k-means/HMM), temporal summary statistics (dwell time, transition probabilities).
- Graph Features: Node strength, efficiency, modularity, rich-club coefficients, MSTs; often fed to classical ML.
- Timeseries Direct Modeling: 1D CNNs/RNNs/LSTMs/Transformers directly on ROI time series; can combine with FC-derived channels.

## Modeling Approaches In Prior Work

- Classical ML on sFC vectors or graph metrics: SVM, Random Forests, Elastic Net; interpretable but limited capacity for nonlinearity.
- CNNs on FC matrices: Leverage local structure and symmetry; variants include BrainNetCNN-style edge-to-edge and edge-to-node filters.
- Sequence models for dFC and raw time series: LSTM/GRU, Temporal CNNs; capture temporal dependencies and dynamic states.
- Graph Neural Networks: Model ROIs as nodes with edges from FC or structural priors; GCN/GAT variants can incorporate node attributes.
- Hybrid Pipelines: dFC windowing + CNN for spatial patterns + LSTM for temporal dynamics; often stronger than single-family models.

## Evaluation Practices And Pitfalls

- Leakage avoidance: Subject-wise splits (GroupKFold/GroupShuffleSplit); window-level augmentation must not mix a subject across folds.
- Metrics: Report accuracy, ROC–AUC, sensitivity/specificity, F1; provide confidence intervals via bootstrapping.
- Statistical Testing: Permutation testing, DeLong test for AUC differences; multiple-comparison control in ablations.
- Calibration: Reliability curves and Brier score; clinically relevant thresholds prioritize sensitivity.
- External Validation: If possible, cross-site testing; otherwise, nested CV and site-stratified evaluation.

## Infant/Toddler-Specific Literature Themes

- Strong age effects: FC patterns rapidly evolve during the first two years; models trained on older children or adults can underperform without adaptation.
- Motion and scan duration: Aggressive motion control and robust estimators are crucial; shorter windows introduce FC estimation bias.
- Emerging work on early ASD risk markers suggests alterations in default mode, salience, and sensorimotor networks, but replication is mixed due to small samples.

## Where Your Current Approach Fits

- Sliding-window augmentation and GroupShuffleSplit match best practice for leakage prevention when using windows.
- 1D CNN on ROI time series ("static CNN") models temporal local patterns directly.
- dFC-CNN leverages windowed correlation matrices, learning spatial connectivity motifs.
- Hybrid CNN+LSTM combines spatial filters (CNN) with temporal integration (LSTM), consistent with contemporary state of the art for dFC sequences.

## Gaps And Opportunities For Novelty

- Pediatric focus: Many deep-learning ASD studies emphasize ABIDE (older cohorts). A carefully designed infant/toddler analysis with robust confound control is a publishable angle.
- Methodological rigor: Few papers combine subject-aware windowing, comprehensive confound adjustment, calibrated probabilities, and interpretable connectivity maps in one pipeline.
- Interpretability: Mapping salient edges/ROIs to neurodevelopmental systems (using Harvard–Oxford labels) strengthens biological plausibility.
- Data efficiency: Demonstrating stable performance under reduced scan lengths and with uncertainty estimates addresses real clinical constraints.

## Recommended Enhancements Toward Publication

- Rigorous validation: Nested subject-wise CV; report mean±CI; add permutation tests and calibration.
- Confound handling: Include age (months), sex, motion, and site as covariates or use harmonization; conduct sensitivity analyses.
- Ablations: Window size/stride; dFC vs sFC vs hybrid; model capacity; regularization; with/without subject-wise standardization.
- External checks: If another cohort is unavailable, perform leave-site-out or simulated domain shift (noise/motion) robustness tests.
- Interpretability: Integrated gradients/Grad-CAM for time series; edge-wise relevance on dFC; summarize top ROIs/edges and network-level patterns.
- Reporting: Learning curves; PR curves; decision thresholds optimizing sensitivity for screening use-cases.

## Ethical And Reproducibility Considerations

- Avoid overclaiming clinical readiness; emphasize exploratory and supportive nature.
- Share code with deterministic seeds; document preprocessing and QC; include scripts to reproduce splits and metrics.

## References (to be completed during manuscript drafting)

- ABIDE I/II datasets for ASD connectomics.
- Reviews on dynamic functional connectivity in neurodevelopment.
- BrainNetCNN and CNNs on FC matrices.
- GNN-based connectome classification.
- Infant/Baby Connectome Project and dHCP methodological notes.
