# Experiment Log

This log summarizes the main modelling stages. The early holdout experiments and final GroupKFold OOF evaluation use different validation designs; compare them as a progression, not as a strictly controlled head-to-head benchmark.

| Stage | Model / method | Evaluation setup | Result | Main takeaway |
|---|---|---|---:|---|
| Baseline | DummyClassifier (`most_frequent`) | Baseline reference | 0.2914 accuracy | Established the majority-class floor |
| Linear | Logistic Regression | Initial stratified holdout | 0.6272 accuracy | Linear boundaries did not capture all relationships |
| Neural | MLPClassifier | Initial stratified holdout | 0.6677 accuracy | Captured some non-linearity but underperformed tree boosting |
| Tree-based | HistGradientBoosting | Initial stratified holdout | 0.7138 accuracy | Better captured non-linear feature interactions |
| Final ensemble | HistGradientBoosting + LightGBM + XGBoost + CatBoost model families | 5-fold GroupKFold grouped by operator, with nested OOF target encoding | 0.72083 OOF accuracy | Combined complementary boosting approaches with group-aware validation |
| Final selected submission | Soft-voting ensemble with class-probability multiplier tuning | Private competition leaderboard | 0.71780 competition score | Final selected result |

## Additional documented findings

- Training data: 279,746 rows and 59 input features.
- Facility JSON combinations: 251,649 distinct strings composed of 144 individual facility codes.
- Test-operator overlap with training operators: 36.0%.
- Test building-coordinate overlap with training coordinates: 87.25%.
- Final feature representation: 292 features.
- Tier 3 recall after probability multiplier tuning: 0.5583, compared with 0.5337 before tuning.

## Before publishing changes

Check each metric against the saved final notebook and permitted leaderboard evidence. Keep leaderboard score, OOF accuracy, and holdout accuracy clearly labelled. Do not add private competition links, data, or credentials.
