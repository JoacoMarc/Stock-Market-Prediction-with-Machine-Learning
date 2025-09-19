import yfinance as yf
from news_analysis import sentiment_analysis
import pandas as pd
import numpy as np


def add_advanced_technical_indicators(stockData):
    """
    Agrega indicadores técnicos avanzados para mejorar las predicciones
    """
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    stockData['RSI_14'] = calculate_rsi(stockData['Close'])
    stockData['RSI_7'] = calculate_rsi(stockData['Close'], 7)
    stockData['RSI_30'] = calculate_rsi(stockData['Close'], 30)
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = stockData['Close'].ewm(span=12).mean()
    ema26 = stockData['Close'].ewm(span=26).mean()
    stockData['MACD'] = ema12 - ema26
    stockData['MACD_Signal'] = stockData['MACD'].ewm(span=9).mean()
    stockData['MACD_Histogram'] = stockData['MACD'] - stockData['MACD_Signal']
    
    # MACD ratios para normalizar
    stockData['MACD_Ratio'] = stockData['MACD'] / stockData['Close']
    stockData['MACD_Signal_Ratio'] = stockData['MACD_Signal'] / stockData['Close']
    
    # Bollinger Bands
    sma20 = stockData['Close'].rolling(window=20).mean()
    std20 = stockData['Close'].rolling(window=20).std()
    stockData['BB_Upper'] = sma20 + (std20 * 2)
    stockData['BB_Lower'] = sma20 - (std20 * 2)
    stockData['BB_Position'] = (stockData['Close'] - stockData['BB_Lower']) / (stockData['BB_Upper'] - stockData['BB_Lower'])
    stockData['BB_Width'] = (stockData['BB_Upper'] - stockData['BB_Lower']) / sma20
    
    # Stochastic Oscillator
    low_min = stockData['Low'].rolling(window=14).min()
    high_max = stockData['High'].rolling(window=14).max()
    stockData['Stoch_K'] = 100 * (stockData['Close'] - low_min) / (high_max - low_min)
    stockData['Stoch_D'] = stockData['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    stockData['Williams_R'] = -100 * (high_max - stockData['Close']) / (high_max - low_min)
    
    # Average True Range (ATR) - Volatilidad
    high_low = stockData['High'] - stockData['Low']
    high_close = np.abs(stockData['High'] - stockData['Close'].shift())
    low_close = np.abs(stockData['Low'] - stockData['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    stockData['ATR_14'] = true_range.rolling(window=14).mean()
    stockData['ATR_Ratio'] = stockData['ATR_14'] / stockData['Close']
    
    # Commodity Channel Index (CCI)
    typical_price = (stockData['High'] + stockData['Low'] + stockData['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    stockData['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index (MFI)
    typical_price = (stockData['High'] + stockData['Low'] + stockData['Close']) / 3
    money_flow = typical_price * stockData['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
    
    money_ratio = positive_flow / negative_flow
    stockData['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # On-Balance Volume (OBV)
    obv = []
    obv_value = 0
    
    for i in range(len(stockData)):
        if i == 0:
            obv_value = stockData['Volume'].iloc[i]
        else:
            if stockData['Close'].iloc[i] > stockData['Close'].iloc[i-1]:
                obv_value += stockData['Volume'].iloc[i]
            elif stockData['Close'].iloc[i] < stockData['Close'].iloc[i-1]:
                obv_value -= stockData['Volume'].iloc[i]
        obv.append(obv_value)
    
    stockData['OBV'] = obv
    stockData['OBV_Ratio'] = stockData['OBV'] / stockData['OBV'].rolling(20).mean()
    
    # Momentum indicators
    stockData['Momentum_5'] = stockData['Close'] / stockData['Close'].shift(5) - 1
    stockData['Momentum_10'] = stockData['Close'] / stockData['Close'].shift(10) - 1
    stockData['Momentum_20'] = stockData['Close'] / stockData['Close'].shift(20) - 1
    
    # Rate of Change (ROC)
    stockData['ROC_5'] = stockData['Close'].pct_change(periods=5) * 100
    stockData['ROC_10'] = stockData['Close'].pct_change(periods=10) * 100
    stockData['ROC_20'] = stockData['Close'].pct_change(periods=20) * 100
    
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
    
    # Agregar indicadores técnicos avanzados
    stockData = add_advanced_technical_indicators(stockData)
    
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
    
    # Agregar indicadores técnicos avanzados a los predictores
    technical_indicators = [
        "RSI_14", "RSI_7", "RSI_30",
        "MACD_Ratio", "MACD_Signal_Ratio", "MACD_Histogram", 
        "BB_Position", "BB_Width",
        "Stoch_K", "Stoch_D",
        "Williams_R",
        "ATR_Ratio",
        "CCI",
        "MFI",
        "OBV_Ratio",
        "Momentum_5", "Momentum_10", "Momentum_20",
        "ROC_5", "ROC_10", "ROC_20"
    ]
    
    # Solo agregar indicadores que existen y no tienen NaN
    for indicator in technical_indicators:
        if indicator in stockData.columns and not stockData[indicator].isna().all():
            predictors.append(indicator)
    
    stockData=stockData.dropna()
    return  stockData, predictors

