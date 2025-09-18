import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from data_from_stock import getDataFromStock, setDataForTraining
from stock_analysis import backtest 


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 2)
model=RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
sp500_data=getDataFromStock("^GSPC", "max")
sp500_data,predictors=setDataForTraining(sp500_data)
predictions=backtest(sp500_data,model,predictors)
print(predictions["Predictions"].value_counts())
print("Precision score: {}".format(precision_score(predictions["Target"], predictions["Predictions"])))