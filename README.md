# Task 1 - Understanding Credit Risk

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The **Basel II Accord** fundamentally shifted regulatory expectations by introducing the _Internal Ratings-Based (IRB)_ approach, which permits institutions to use internal models to estimate key risk parameters: **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**.

This confers flexibility—but also imposes rigorous requirements:

- **Transparency & Documentation**: Models must be fully documented, with clear rationale for variable selection, transformation (e.g., Weight of Evidence binning), and calibration. Regulators (e.g., HKMA, Fed) routinely audit model development and validation processes.
- **Governance & Validation**: Independent model validation, back-testing against realized defaults, and stress testing are mandatory. As emphasized in the World Bank (2020) and HKMA guidance, _interpretability underpins auditability_.
- **Stability & Consistency**: Models must demonstrate performance stability over time and across economic cycles. Black-box models may achieve high discrimination today but fail to satisfy long-term regulatory expectations.

Thus, while predictive performance is important, **interpretability is a regulatory necessity**—not a luxury. A well-documented logistic regression with WoE transformations aligns with Basel II principles and facilitates supervisor trust.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In early-stage portfolios, thin-file borrowers, or new digital lending products, _observed defaults are rare, delayed, or absent_. As noted in the HKMA and World Bank reports, lenders often resort to **behavioral proxies** for default, such as:

- ≥30/60/90 Days Past Due (DPD),
- Missed payment counts,
- Account suspension or write-off flags,
- Internal delinquency grades.

While proxies enable timely model development, they introduce material risks:

- **Construct Mismatch**: A proxy (e.g., 60 DPD) may not equate to _economic default_ (e.g., loss > 0%). Some borrowers cure; others strategically delay payments.
- **Calibration Error**: PD estimates derived from proxy events must be scaled to reflect _long-run default frequencies_ (per Basel II). Without proper scaling, capital reserves may be misestimated.
- **Regulatory Scrutiny**: Supervisors require evidence that proxy-based PDs are conservative, validated, and periodically reconciled with eventual default outcomes.

> Best practice: Treat proxy-based models as _interim solutions_, with a clear roadmap to recalibrate once true default data matures.

---

### 3. What are the key trade-offs between using a simple, interpretable model (e.g., Logistic Regression with WoE) versus a complex, high-performance model (e.g., Gradient Boosting) in a regulated financial context?

| Criterion                       | Logistic Regression + WoE                                                                                                 | Gradient Boosting (XGBoost/LightGBM)                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Regulatory Acceptance**       | High — Industry standard for IRB scorecards; explicitly endorsed in HKMA & World Bank guidelines.                         | Conditional — Requires extensive validation, explainability (e.g., SHAP), and may only be acceptable as a _challenger_ model. |
| **Interpretability**            | Transparent coefficients, monotonic WoE bins, intuitive reason codes for adverse actions (e.g., Fair Lending compliance). | Low — Complex interactions hinder root-cause analysis; post-hoc explanations may not satisfy auditors.                        |
| **Stability & Robustness**      | High — Binning reduces noise; less sensitive to outliers or small data shifts. Performs reliably out-of-time.             | Moderate — Prone to overfitting on sparse segments; performance may degrade rapidly in stress/regime shifts.                  |
| **Implementation & Monitoring** | Low operational risk — Easy to deploy, monitor bin drift, and adjust manually.                                            | Higher overhead — Requires MLOps pipelines, drift detection on raw features, and model versioning.                            |
| **Predictive Power (AUC/Gini)** | Good — Often sufficient for business use; ceiling due to linearity.                                                       | Typically higher — Captures non-linearities and high-order interactions.                                                      |

> **Recommendation**: Start with a **WoE-based logistic regression** as the production model for compliance and control. Use **tree-based models for insight generation** (e.g., uncovering non-linear risk patterns to inform bin design), and consider ensembles or stacking _only_ after rigorous regulatory consultation and validation.
