# um-game-playing-strength

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Kaggle Workspace is env for kaggle competition.

[Leaderboard](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/leaderboard) | [Discussion](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion?sort=published)

## Solution

## Experiment

| EXP_ID | Local CV | Public LB | Note |
| :---: | :---: | :---: | :--- |
| 1 | 0.43919718875979596 | 0.456 | simple baseline |
| 2 | 0.43453696524665734 | 0.446 | refactor |
| 3 | 0.43026924496271557 | 0.446 | avg xgb and lgbm |
| 4 | 0.42618337240756193 | 0.442 | clip target -1.0 ~ 1.0 |
| 5 | 0.42616738571265820 | --- | ensemble with ridge |
| 6 | 0.4213981145655368 | 0.434 | add catboost |
| 7 | 0.4194373177906728 | 0.434 | ensemble weighted models |
| 8 | 0.4170520075148410 | 0.436  | tuning hypara of lgbm and cat |
| 9 | 0.41748504610705306 | 0.435  | tuning hypara of lgbm_weighted and cat_weightd |
| 10 | 0.40692605401488136 | 0.425  | add reverse train to features |
| 11 | 0.40473969633293044 | 0.426  | add model of lgbm_drop |
| 12 | 0.40679631169455976 | 0.426  | remove early stopping |
| 13 | 0.40175697223964435 | 0.429  | add is_raw as a feature |

## Not worked for me

- Target Encoding of Agent Features
- k=4,7 of GroupKFold
- Use StratifiedGroupKFold instead of GroupKFold
- Rounding prediction values
  - utility_agent1 is only about 50 values
- Stacking only when prediction value is -1.0 < pred < 1.0
- 2 stage modeling
  - fist stage: classify target value is -1.0 < target < 1.0 or -1.0 or 1.0
  - sencond stage: regression target value is -1.0 < target < 1.0
- Ensemble Simple NN Model and TabNetRegressor
  - i can't get good cv score, it's about 0.65

## Idea

- model
  - hyperparameter tuning of cat and lgbm again
