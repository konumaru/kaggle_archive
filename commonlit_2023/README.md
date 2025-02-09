# CommonLit2023

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Leaderboard](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/leaderboard) | [Discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion?sort=published)

## Solution

- feature engineering
  - feature list
    - text length
    - word count
    - sentence count
    - quoted sentence count
    - consec tive_dots count
    - word overlap count
    - spell miss count, SpellChecker
  - text embedding
    - deberta v3
  - target encoding of content and wording
    - groupby word count (rounded to 5)
    - sentence count (clip between 1 and 20)
- cv strategy
  - Group K-Fold
    - k=4
    - group=prompt_id
- model
  - first statge model
    - fine tuned deberta v3 base (cv=0.55)
      - inputs: prompt_question, text
  - second stage models
    - XGBoost (cv=0.5222257984694146)
      - inputs: first stage output, text stats feature
    - LightGBM (cv=0.5239820553996042)
      - inputs: first stage output, text stats feature
  - ensemble
    - simple average of second stage models.

## Experiments

| EXP_ID | Local CV | Public LB | Note |
| :---: | :---: | :---: | :--- |
| 1 | 0.6687954845101823 | 0.599 | rf with simple text feature |
| 2 | 0.5148155805419965 | - | add feature of debertav3 text embeddings |
| 3 | 0.4903529269444470 | 0.509 | change model from rf to xgb |
| 4 | 0.4899955738087213 | - | add featrue of debertav3 prompt embeddings |
| 5 | 0.4785185756657641 | - | add feature of overlap word and co-occur words |
| 6 | 0.4759433370779221 | - | add feature of tri-gram co-occur words |
| 7 | 0.4737618975123431 | - | change xgb n_estimatoers param 500 to 800 |
| 8 | 0.4744999729694380 | 0.479 | rm featrue of debertav3 prompt embeddings |
| 9 | 0.5576348008005831 | 0.478 | **change kfold to group kfold** |
| 10 | 0.5572727558437666 | - | add feature of spell_miss_count |
| 11 | 0.5560561772865491 | 0.479 | add feature of quotes_count |
| 12 | 0.5451717268584183 | 0.559 | only finetuned deberta base |
| 13 | 0.5168956770838019 | 0.491 | stacking xgb on deberta |
| 14 | 0.5162055570275468 | - | ensenble lgbm |
| 15 | 0.5148750859363870 | 0.465 | add feature of target encoding |
| 16 | 0.5157331434893387 | 0.467 | refactoring create feature process |
| 17 | 0.5114196777076987 | 0.470 | add feature of wv simirality of prompt text and text |
| 18 | 0.5097619287866334 | - | add feature of some text stats |
| 19 | 0.509488368861999 | 0.467 | add feature of glove vec simirality |

## Not worked for me

- fine tuned roberta base (cv=0.5809940545327481) as first stage model
  - inputs: prompt_question, text
- text averaged word2vec
- average and median of word length
- text length per sentence
- training deberta each other prompt and summary
