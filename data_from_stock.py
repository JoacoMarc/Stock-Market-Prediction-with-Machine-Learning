import yfinance as yf


def getDataFromStock(stockName, period):
    stock= yf.Ticker(stockName)
    stockData = stock.history(period=period) 
    return stockData


def setDataForTraining(stockData):
    predictors=["Open", "High", "Low", "Close", "Volume"]
    stockData=stockData.loc["1993-01-01":].copy()
    stockData=stockData.drop("Dividends", axis=1, errors='ignore')  
    stockData=stockData.drop("Stock Splits", axis=1, errors='ignore')  
    stockData["Tomorrow"]=stockData["Close"].shift(-1)
    stockData["Target"]= (stockData["Tomorrow"] > stockData["Close"]).astype(int)
    tendencyDays = [2,5,60,250,1000]
    for tendencyDay in tendencyDays:
        rollingAverage=stockData.rolling(tendencyDay).mean()
        ratioColumn="Close_Ratio_" + str(tendencyDay)
        stockData[ratioColumn]=stockData["Close"]/rollingAverage["Close"]
        trendColumn="Trend_" + str(tendencyDay)
        stockData[trendColumn]=stockData.shift(1).rolling(tendencyDay).sum()["Target"]
        predictors.append(ratioColumn)
        predictors.append(trendColumn)
    stockData=stockData.dropna()
    return  stockData, predictors

