from sklearn.metrics import precision_score
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 2)
model=RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

def getDataFromStock(stockName, period):
    stock= yf.Ticker(stockName)
    stockData = stock.history(period=period) 
    stockData=stockData.drop("Dividends", axis=1, errors='ignore')  
    stockData=stockData.drop("Stock Splits", axis=1, errors='ignore')  
    return stockData

def create_graph(height,lenght, data, title, color, linewidth, xlabel, ylabel):
    plt.figure(figsize=(lenght, height))

    data['Close'].plot(title, 
                    color,
                    linewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def setDataForTraining(stockData):
    stockData["Tomorrow"]=stockData["Close"].shift(-1)
    stockData["Target"]= (stockData["Tomorrow"] > stockData["Close"]).astype(int)
    return stockData.loc["1990-01-01":].dropna() 

def trainModel(stockData):
    train= stockData.iloc[:-100]
    test= stockData.iloc[-100:]
    predictors=["Open", "High", "Low", "Close", "Volume"]
    model.fit(train[predictors], train["Target"])
    preds=model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test["Target"], preds], axis=1)
    return combined 

sp500_data=getDataFromStock("^GSPC", "max")
sp500_data=setDataForTraining(sp500_data)
preds, accuracy = trainModel(sp500_data)
print(preds)
print(f"Model accuracy: {accuracy}")



#create_graph(6,12, sp500_data, "S&P 500 Historical Data", 
    #"blue", 1, "Date", "Close Price")
