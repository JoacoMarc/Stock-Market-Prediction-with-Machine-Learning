import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def predict(train,test, predictors,model):
    model.fit(train[predictors], train["Target"])
    preds=model.predict_proba(test[predictors])[:,1]
    preds = [1 if x >= 0.6 else 0 for x in preds]
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(stockData, model, predictors, start=2500, step=250 ):
    all_predictions=[]
    for i in range (start, stockData.shape[0], step):
        train=stockData.iloc[0:i].copy()
        test=stockData.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


