# Credit Risk Probability Model for Alternative Data


---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord fundamentally restructures how financial institutions measure and manage credit risk. Under Basel II's Internal Ratings-Based (IRB) approach, banks are permitted to use their own internal models to estimate key risk parameters — Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) — to calculate regulatory capital requirements. However, this permission comes with a critical obligation: the models must be **transparent, explainable, and rigorously documented**.

This regulatory mandate has several direct consequences for our modeling approach:

**Interpretability as a regulatory requirement, not a preference.** Basel II Pillar 2 (Supervisory Review) requires that regulators can scrutinize and validate a bank's internal models. A black-box model — even one with superior predictive accuracy — cannot satisfy this requirement because regulators must be able to assess the logic driving risk classifications. A Logistic Regression model with Weight of Evidence (WoE) encoding, for instance, allows a supervisor to inspect exact coefficient magnitudes and understand which behavioral features are driving the risk score. Gradient Boosting models, while often more accurate, require supplemental explainability tools (e.g., SHAP values) to meet this bar.

**Documentation as a first-class deliverable.** Basel II mandates a "use test" — the internal model must be genuinely used in credit decision-making, not built solely for regulatory compliance and shelved. This requires comprehensive documentation covering: data lineage, variable selection rationale, model validation results, performance monitoring procedures, and known limitations. Every modeling choice in this project (proxy variable design, feature selection, threshold setting) must be traceable and defensible.

**Model monitoring and stability requirements.** Basel II requires banks to demonstrate that their models remain accurate over time through backtesting and benchmarking. This shapes our MLOps architecture: experiment tracking via MLflow, versioned artifacts, and a CI/CD pipeline ensure that model updates are controlled, reproducible, and auditable.

**Auditability of the training pipeline.** Setting `random_state` on every stochastic operation (train/test splits, clustering, model training) and versioning the processed dataset are not just good engineering practices — they are prerequisites for meeting Basel II's reproducibility expectations. Without them, results cannot be replicated during a regulatory audit.

In summary, Basel II doesn't merely constrain model complexity — it elevates documentation, interpretability, and governance to the same priority level as predictive performance.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

**Why a proxy variable is necessary:**

The Xente eCommerce dataset contains transaction-level behavioral data — purchase history, frequency, recency, monetary value, product categories, channels — but it contains **no historical loan performance records and no ground-truth default labels**. A credit scoring model is fundamentally a supervised learning problem: it requires a target variable (defaulted vs. not defaulted) to learn from. Without a direct default label, we cannot train a classifier using standard supervised methods.

The proxy variable bridges this gap by constructing a synthetic approximation of credit risk from observable behavioral signals. The underlying assumption is that **customers who are financially disengaged, transact infrequently, and spend minimally are structurally similar to borrowers who fail to repay loans** — they exhibit low financial commitment and reduced capacity or willingness to service debt. By applying RFM (Recency, Frequency, Monetary) analysis and clustering customers into behavioral segments, we can label the most disengaged cluster as "high risk" (proxy for likely default) and all others as "low risk."

**Business risks introduced by proxy-based prediction:**

The proxy variable is a **modeling assumption, not ground truth**. This distinction carries significant and unavoidable business risks:

- **Construct validity risk.** Behavioral disengagement on an eCommerce platform does not necessarily translate to credit default behavior. A customer who shops infrequently might simply be a disciplined saver — not a credit risk. The proxy may systematically misclassify entire demographic groups, introducing both financial and ethical exposure.

- **Label noise and its amplification.** Every mislabeled training example degrades model quality. Unlike true default labels — which are binary facts (a borrower either defaulted or did not) — proxy labels are probabilistic inferences. Errors in the proxy propagate through the entire model, producing a classifier that learns an imperfect signal.

- **Regulatory defensibility.** Using a proxy target without a clear audit trail and theoretical justification puts the bank at regulatory risk. Basel II requires that risk parameters reflect actual default experience. A proxy-based PD estimate must be framed carefully in regulatory submissions — as a behavioral risk indicator — with explicit acknowledgment of its limitations and a validation plan against actual loan outcomes once BNPL lending begins.

- **Feedback loop risk.** If the proxy-based model is used to approve or reject BNPL applicants, the resulting loan portfolio may not include sufficient defaulters to validate or retrain the model accurately. Only customers the model judged "low risk" will receive loans, creating survivorship bias in subsequent model validation.

- **Temporal instability.** RFM-based clusters are a snapshot of customer behavior at a point in time. If the eCommerce platform's user base, product mix, or pricing strategy shifts, the behavioral patterns underlying the proxy may change — invalidating the risk labels without triggering any explicit model failure signal.

**Mitigation strategy:** Document all proxy assumptions explicitly. Establish a validation roadmap: once the BNPL product launches and actual repayment data accumulates (typically 12–18 months), recalibrate the model against true default labels and retire the proxy.

---

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

The choice between a Logistic Regression with WoE encoding and a Gradient Boosting model (e.g., XGBoost or LightGBM) represents one of the central tensions in regulated credit scoring. Neither approach dominates across all evaluation criteria.

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| **Predictive performance** | Moderate — linear decision boundary may miss complex interactions | High — captures non-linear patterns and feature interactions automatically |
| **Interpretability** | High — coefficients directly quantify each feature's log-odds contribution; scorecards can be hand-checked | Low — hundreds of trees interact; requires SHAP or LIME for post-hoc explanation |
| **Regulatory acceptance** | High — well-established in Basel II IRB contexts; auditors and risk committees understand it | Moderate to low — requires supplemental documentation and explainability tooling |
| **Development time** | Low — WoE binning is manual but well-understood; model is stable | High — hyperparameter search is computationally intensive and result is sensitive to data changes |
| **Stability / monotonicity** | High — WoE encoding enforces monotonic risk ordering by design; model behavior is predictable under input shifts | Low — small changes in training data can produce large, unpredictable score shifts |
| **Handling imbalanced classes** | Requires explicit weighting or resampling | Handles imbalance more robustly via built-in `scale_pos_weight` and boosting mechanics |
| **Maintenance burden** | Low — model logic is transparent; recalibration is straightforward | High — model is a black box; debugging score drift requires feature importance analysis |

