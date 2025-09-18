import yfinance as yf
from news_analysis import sentiment_analysis
import pandas as pd

def getDataFromStock(stockSymbol, period):
    stock= yf.Ticker(stockSymbol)
    stockData = stock.history(period=period) 
    return stockData


def setDataForTraining(stockSymbol, stockName=None):
    startYear = '1990-01-01'
    stockData = yf.download(stockSymbol, start=startYear, end=None)
    
    # Si las columnas son MultiIndex, aplanarlas manteniendo solo el nombre de la columna
    if isinstance(stockData.columns, pd.MultiIndex):
        stockData.columns = [col[0] for col in stockData.columns]
    
    # Limpiar nombres de columnas (remover espacios)
    stockData.columns = [col.replace(' ', '_') if isinstance(col, str) else col for col in stockData.columns]
    
    stockData=stockData.drop("Dividends", axis=1, errors='ignore')  
    stockData=stockData.drop("Stock_Splits", axis=1, errors='ignore')  
    
    # Agregar columnas de sentiment con valores neutros por defecto
    stockData["Sentiment_Positive"] = 0.33
    stockData["Sentiment_Negative"] = 0.33
    stockData["Sentiment_Neutral"] = 0.34
    
    stockData["Tomorrow"]=stockData["Close"].shift(-1)
    stockData["Target"]= (stockData["Tomorrow"] > stockData["Close"]).astype(int)
    
    predictors = []  # Inicializar la lista de predictores
    tendencyDays = [2,5,60,250,1000]
    for tendencyDay in tendencyDays:
        rollingAverage=stockData.rolling(tendencyDay).mean()
        ratioColumn="Close_Ratio_" + str(tendencyDay)
        stockData[ratioColumn]=stockData["Close"]/rollingAverage["Close"]
        trendColumn="Trend_" + str(tendencyDay)
        stockData[trendColumn]=stockData["Target"].shift(1).rolling(tendencyDay).sum()
        predictors.append(ratioColumn)
        predictors.append(trendColumn)
    
    # Agregar las columnas de sentiment a los predictores
    predictors.extend(["Sentiment_Positive", "Sentiment_Negative", "Sentiment_Neutral"])
    
    stockData=stockData.dropna()
    return  stockData, predictors

