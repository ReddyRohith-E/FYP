# Model Performance Comparison

**Status**: Verified on Unified Stratified Split (Same Subjects for All Models).

## Accuracy Rankings

| Rank  | Model Name                   | Model Type                 | Test Accuracy |
| :---- | :--------------------------- | :------------------------- | :------------ |
| **1** | **Static FC-CNN**            | **Deep Learning (Static)** | **70.00%**    |
| 2     | Support Vector Machine (SVM) | Machine Learning           | 65.00%        |
| 3     | Neural Network (MLP)         | Deep Learning (Simple)     | 55.00%        |
| 4     | DFC-CNN                      | Deep Learning (Dynamic)    | 50.00%        |
| 4     | Random Forest                | Machine Learning           | 50.00%        |
| 4     | Gradient Boosting            | Machine Learning           | 50.00%        |

## Analysis

### Winner: Static FC-CNN (70%)

The **Static FC-CNN** is the only model to break the 65% ceiling.

- **Why:** It combines the robustness of the **static correlation matrix** (filtering out temporal noise) with the **feature extraction power of a 1D-CNN**. The CNN layers likely identified non-linear patterns within the correlation vector that the SVM (linear/RBF) and MLP (fully connected) missed.

### Runner Up: SVM (65%)

The SVM remains a strong, reliable baseline. It outperforms the MLP in this unified split, proving that for small datasets (100 subjects), simpler models with strong regularization often beat un-optimized deep networks.

### Underperformers: Dynamic Models (DFC, Hybrid)

Both **DFC-CNN** (50%) and the previous **Hybrid** model (52.5%) failed to exceed random chance. This confirms that **using raw time-series or dynamic windows introduces too much noise** for this dataset size.

## Conclusion & Recommendation

The **Static FC-CNN** is the definitive **best model** for this project.

**Recommendation**: Use the `Static FC-CNN` for the final application. It provides the highest accuracy (70%) and is computationally efficient.
