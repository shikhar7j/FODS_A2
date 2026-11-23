# CS F320: Foundations of Data Science — Assignment 2
## Principal Component Analysis Report

**Submission Deadline:** 23:55 Hrs, 23rd Nov 2026  
**Dataset:** Wine dataset (`wine_train.csv`: $N=124$, `wine_test.csv`: $N_{\text{test}}=54$, $D=13$ features, 3 classes)

---

## P1: PCA from First Principles

### 1. Eigenvalues Output
As required, the 13 eigenvalues of the sample covariance matrix $S$ computed on standardized training data (in descending order, formatted to 4 decimal places):

```text
lambda_1 = 4.6450
lambda_2 = 2.4972
lambda_3 = 1.4309
lambda_4 = 0.9426
lambda_5 = 0.9065
lambda_6 = 0.6944
lambda_7 = 0.5922
lambda_8 = 0.3140
lambda_9 = 0.2653
lambda_10 = 0.2570
lambda_11 = 0.2191
lambda_12 = 0.1413
lambda_13 = 0.0947
```

### 2. Matrix Norm and Trace Properties
- **Frobenius norm of orthonormality residual:** $\|U^\top U - I\|_F = \mathbf{3.7\text{e-}15}$
- **Trace identity absolute gap:** $|\text{tr}(S) - \sum_{j} \lambda_j| = \mathbf{1.8\text{e-}15}$
- **Sample covariance trace:** $\text{tr}(S) = \mathbf{13.0000}$ (since each standardized feature has $s_j^2 = 1$, $\text{tr}(S) = \sum_{j=1}^{13} 1 = 13$).

### 3. Verification with `sklearn.decomposition.PCA`
Using `PCA(n_components=13, svd_solver="full")` and applying the C4 sign convention to the rows of `components_`:
- **Maximum absolute difference between our eigenvalues and `explained_variance_`:** $\mathbf{6.2\text{e-}15}$
- **Maximum absolute difference between our $U$ and `components_.T`:** $\mathbf{8.9\text{e-}15}$
- Both differences are well below the threshold of $10^{-8}$, verifying exact agreement per C6.

### 4. Loadings Table and Top Features

| Feature Name | $u_1$ (PC1 Loading) | $u_2$ (PC2 Loading) |
| :--- | :---: | :---: |
| alcohol | 0.1576 | 0.4941 |
| malic_acid | -0.2524 | 0.1671 |
| ash | -0.0166 | 0.3358 |
| alcalinity_of_ash | -0.2351 | -0.0230 |
| magnesium | 0.1429 | 0.2817 |
| total_phenols | 0.3899 | 0.0716 |
| flavanoids | 0.4260 | 0.0057 |
| nonflavanoid_phenols | -0.2868 | 0.0486 |
| proanthocyanins | 0.3086 | 0.0059 |
| color_intensity | -0.0889 | 0.5263 |
| hue | 0.3012 | -0.2840 |
| od280/od315_of_diluted_wines | 0.3778 | -0.1814 |
| proline | 0.2884 | 0.3703 |

- **Three features with largest $|loading|$ on PC1:**
  1. `flavanoids` ($|0.4260|$)
  2. `total_phenols` ($|0.3899|$)
  3. `od280/od315_of_diluted_wines` ($|0.3778|$)
- **Three features with largest $|loading|$ on PC2:**
  1. `color_intensity` ($|0.5263|$)
  2. `alcohol` ($|0.4941|$)
  3. `proline` ($|0.3703|$)

---

## P2: Variance Captured vs. Number of Components

### 1. Component Selection Metrics
- **Elbow criterion:** $k_{\text{elbow}} = \max\{k : r_k \ge 0.10\} = \mathbf{3}$
- **Cumulative variance at elbow:** $R(k_{\text{elbow}}) = R(3) = \mathbf{0.6595}$ ($65.95\%$)
- **Variance thresholds $k_\tau = \min\{k : R(k) \ge \tau\}$:**
  - $k_{0.80} = \mathbf{5}$
  - $k_{0.90} = \mathbf{7}$
  - $k_{0.95} = \mathbf{10}$
- **Kaiser criterion count:** $\lambda_j > \text{tr}(S)/D = 1 \implies \mathbf{3}$ components ($\lambda_1 = 4.6450, \lambda_2 = 2.4972, \lambda_3 = 1.4309 > 1$, while $\lambda_4 = 0.9426 < 1$).

### 2. Table of Variance and Reconstruction Error ($k = 1, \dots, 13$)

| $k$ | $r_k$ | $R(k)$ | $J_{\text{train}}(k)$ | $J_{\text{test}}(k)$ |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3573 | 0.3573 | 8.3550 | 8.5455 |
| 2 | 0.1921 | 0.5494 | 5.8578 | 6.1120 |
| 3 | 0.1101 | 0.6595 | 4.4270 | 4.5980 |
| 4 | 0.0725 | 0.7320 | 3.4844 | 3.8979 |
| 5 | 0.0697 | 0.8017 | 2.5779 | 3.0326 |
| 6 | 0.0534 | 0.8551 | 1.8835 | 2.4541 |
| 7 | 0.0456 | 0.9007 | 1.2913 | 1.8367 |
| 8 | 0.0242 | 0.9248 | 0.9773 | 1.3861 |
| 9 | 0.0204 | 0.9452 | 0.7120 | 1.0353 |
| 10 | 0.0198 | 0.9650 | 0.4550 | 0.7532 |
| 11 | 0.0169 | 0.9819 | 0.2359 | 0.4497 |
| 12 | 0.0109 | 0.9927 | 0.0947 | 0.1726 |
| 13 | 0.0073 | 1.0000 | 0.0000 | 0.0000 |

### 3. Numerical Verification of Identity
- $\max_{k=1,\dots,13} |J_{\text{train}}(k) - \sum_{j>k} \lambda_j| = \mathbf{1.3\text{e-}15} < 10^{-8}$, verifying exact agreement per C6.

### 4. Written Paragraph Answer (P2)
> $k_{\text{elbow}} = 3$ agrees with the Kaiser criterion count ($k = 3$, since $\lambda_3 = 1.4309 > 1$ while $\lambda_4 = 0.9426 < 1$), but is more parsimonious than the cumulative thresholds $k_{0.80} = 5$, $k_{0.90} = 7$, and $k_{0.95} = 10$. Retaining $k_{\text{elbow}} = 3$ captures $R(3) = 65.95\%$ of total variance, which reflects the steep initial drop off before the tail but may under-represent variance if $\ge 80\%$ retention is required. Algebraically, for any point $z_n$, increasing $k$ removes the term $(u_{k+1}^\top z_n)^2 \ge 0$ from the residual sum $\sum_{j>k} (u_j^\top z_n)^2$, so $\|z_n - \hat{z}_n(k+1)\|^2 \le \|z_n - \hat{z}_n(k)\|^2$. Averaging over test points guarantees that $J_{\text{test}}(k)$ is non-increasing in $k$, which is confirmed by our plot where $J_{\text{test}}(k)$ decreases monotonically from 8.5455 down to 0.

---

## P3: Effect of Preprocessing, and the Two-Dimensional Projection

### 1. Preprocessing Variants Comparison

| Variant | $r_1$ | $r_2$ | $r_1 + r_2$ | Top PC1 Feature | 5-NN Test Accuracy |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **V1 (Centred only)** | 0.9978 | 0.0021 | 0.9998 | proline | 0.7222 |
| **V2 (Min-max)** | 0.4271 | 0.1841 | 0.6113 | flavanoids | 0.9815 |
| **V3 (Z-score)** | 0.3573 | 0.1921 | 0.5494 | flavanoids | 0.9444 |

### 2. Three Largest Raw Sample Variances (V1)
1. `proline`: $\mathbf{94906.7109}$
2. `magnesium`: $\mathbf{232.0715}$
3. `alcalinity_of_ash`: $\mathbf{11.4713}$

### 3. Written Paragraph Answer (P3)
> Variant V1 applies centering without scaling, causing the feature with the largest scale (`proline`, sample variance $\approx 94906.71$) to dominate the objective $u^\top S u$ and align PC1 almost entirely with `proline`, yielding $r_1 = 99.78\%$. In contrast, V3 standardises all features to unit variance, distributing variance across features so $r_1$ is only $35.73\%$. While V1 captures $99.98\%$ of total variance in two components versus $54.94\%$ for V3, V1 yields a much lower 5-NN test accuracy ($72.22\%$ vs. $94.44\%$). This demonstrates that high total variance does not necessarily reflect discriminative class structure; large raw variance often reflects arbitrary measurement units rather than separability. Therefore, V3 (Z-score standardisation) should be used before PCA so features contribute based on correlation structure rather than measurement scale.

---

## P4: PCA versus Feature Selection for a Downstream Classifier

### 1. Baseline Test Accuracy
- Multinomial Logistic Regression on all 13 standardised training features: **0.9815** (53/54 test samples correct).

### 2. Downstream Classifier Test Accuracy Table

| $k$ | PCA (A) | Filter (B) | Wrapper (C) |
| :---: | :---: | :---: | :---: |
| 1 | 0.8704 | 0.8519 | 0.8519 |
| 2 | 0.9444 | 0.9074 | 0.8889 |
| 3 | 0.9815 | 0.9074 | 0.8704 |
| 5 | 1.0000 | 0.9815 | 0.9074 |
| 8 | 0.9815 | 0.9815 | 0.9630 |

### 3. F-score Ranking of all 13 Features (Best First)
1. `flavanoids`: 187.6858
2. `proline`: 149.4335
3. `od280/od315_of_diluted_wines`: 126.6191
4. `alcohol`: 100.3464
5. `color_intensity`: 83.9793
6. `hue`: 72.2673
7. `total_phenols`: 59.2558
8. `alcalinity_of_ash`: 26.0528
9. `malic_acid`: 22.5828
10. `proanthocyanins`: 19.9799
11. `nonflavanoid_phenols`: 17.9053
12. `ash`: 11.5760
13. `magnesium`: 7.1124

### 4. Feature Subsets for $k=3$ and $k=8$
- **$k=3$:**
  - **Filter (B):** `['flavanoids', 'proline', 'od280/od315_of_diluted_wines']`
  - **Wrapper (C):** `['flavanoids', 'alcohol', 'alcalinity_of_ash']`
  - **Shared features:** **1** (`flavanoids`)
- **$k=8$:**
  - **Filter (B):** `['flavanoids', 'proline', 'od280/od315_of_diluted_wines', 'alcohol', 'color_intensity', 'hue', 'total_phenols', 'alcalinity_of_ash']`
  - **Wrapper (C):** `['flavanoids', 'alcohol', 'alcalinity_of_ash', 'proanthocyanins', 'magnesium', 'ash', 'hue', 'proline']`
  - **Shared features:** **5** (`flavanoids`, `alcohol`, `alcalinity_of_ash`, `hue`, `proline`)

### 5. Smallest $k$ Reaching Accuracy within One Test Sample of Baseline
Threshold: $\ge \text{baseline} - 1/54 = 0.9815 - 0.0185 = \mathbf{0.9630}$ (at most one more test error than baseline):
- **PCA:** $k = \mathbf{3}$ (accuracy $0.9815 \ge 0.9630$)
- **Filter:** $k = \mathbf{5}$ (accuracy $0.9815 \ge 0.9630$)
- **Wrapper:** $k = \mathbf{8}$ (accuracy $0.9630 \ge 0.9630$)

### 6. Written Paragraph Answer (P4)
> The baseline test accuracy on all 13 features is $0.9815$ (53/54), giving a tolerance threshold of $\ge 0.9815 - 1/54 = 0.9630$ (52/54); the smallest $k$ reaching this is $k=3$ for PCA ($0.9815$), $k=5$ for Filter ($0.9815$), and $k=8$ for Wrapper ($0.9630$). Although PCA never sees class labels while Filter and Wrapper do, PCA unexpectedly achieves the highest test accuracy across most $k$ (reaching $1.0000$ at $k=5$). This occurs because the principal axes of maximum variance in the Wine dataset naturally align with between-class separation, whereas greedy forward selection risks overfitting the training folds. However, with only 54 test samples, a difference of 1 test error represents an accuracy step of $\approx 1.85\%$, meaning differences between the methods fall within normal sampling variability and do not permit high confidence in their relative ranking.
