# Model Card

## Model Purpose

The draft model ranks which prospect a team is most likely to select at a given slot. It is a behavioral ranking model, not a pure "future NHL value" predictor.

Primary code:

- [features.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/ml/features.py)
- [train.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/ml/train.py)
- [predict.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/ml/predict.py)
- [backtest.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/ml/backtest.py)

## Model Type

- Algorithm: XGBoost LambdaMART ranker
- Task: learning-to-rank
- Training target: the actual prospect selected at each historical draft slot should rank above the players who were still available

## Inputs

The model uses several groups of features:

- Prospect quality:
  - CSS rank normalization
  - points per game
  - age at draft
  - league context
- Team / board context:
  - pick number
  - position scarcity
  - board state
  - team positional draft state
- GM tendency context:
  - position preference
  - league preference
  - nationality preference
  - archetype

## Outputs

The model returns a score for each candidate prospect at a specific pick. Higher score means the model believes the prospect is more plausible for that team and slot.

The app also surfaces:

- SHAP explanations for feature influence
- conformal calibration metadata
- prediction-set membership for UI confidence labels

## Evaluation

Evaluation is temporal, not random.

- Single-year held-out backtests train on years up to a cutoff and evaluate on a future year
- Multi-year backtests repeat that process across several draft classes
- Baseline comparisons are included against:
  - CSS best-available
  - PPG best-available
  - uniform random

Primary metrics:

- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- MRR

## Limitations

- The model predicts draft behavior, not long-term player value
- Historical GM behavior can encode bias or stale preferences
- Prospect data quality depends on ingestion freshness and coverage
- Late-round drafting is noisier than early-round drafting
- Some picks are intentionally surprising and should not always rank highly under the model

## Intended Use

Good uses:

- draft simulation
- comparing heuristic baselines to model behavior
- exploring team/GM draft tendencies
- generating explainable, data-backed draft recommendations

Not ideal as:

- a standalone long-term player success model
- a replacement for actual scouting
- a betting or financial decision engine
