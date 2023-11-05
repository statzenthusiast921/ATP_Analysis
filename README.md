# ATP_Analysis


### Description

The purpose of this project is to:
- build model(s) to predict winners of matches


### Data

The data used for this analysis was taken from this Kaggle page: [thetvdb.com](https://www.kaggle.com/datasets/sijovm/atpdata/data)


### Challenges
- It was a very difficult task to produce a model that did not either have a lousy training/test accuracy or significantly overfit.  The best results came from filtering the data to all matches from 2010 and on and breaking the data out by surface and fitting individual models per surface.  Yet, the highest accuracy I was able to achieve was still only approximately 65%.
