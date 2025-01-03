# ATP Analysis


### Description

The purpose of this project was to:
- build model(s) to predict winners of matches based on in-game player statistics
- practice ensemble modeling
- practice with the [SHAP](https://shap.readthedocs.io/en/latest/) library
- practice working with parquet files


### Data

The data used for this analysis was taken from this [Kaggle](https://www.kaggle.com/datasets/sijovm/atpdata/data) page.  It includes only mens matches from 1968 through 2022.  Match statistics were not available until 1991.


### Challenges
- Since there were no in-game statistics collected (# aces, % of 1st serve in, etc.) before 1991, I decided to filter those matches out from the model.  Prior to this decision, I made an attempt at creating a fully observed dataset by imputing pre-1991 values based on post 1991 values.  However, any model that I produced had a lousy training/test accuracy or significantly overfit.  The best results came from breaking the data out by surface and fitting individual models per surface.  Yet, the highest accuracy I was able to achieve was still only approximately 65%.  Once I removed the pre-1991 values, the average accuracy rose to approximately 75%.
