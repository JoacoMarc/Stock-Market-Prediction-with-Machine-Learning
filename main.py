import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from data_from_stock import getDataFromStock, setDataForTraining
from stock_analysis import backtest 
from news_analysis import clear_sentiment_cache

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 2)

# Limpiar cache de noticias al inicio
clear_sentiment_cache()

# Configuración
stockSymbol = input("Ingrese el símbolo bursátil (por ejemplo, AAPL, MSFT, GOOG): ").strip().upper()
stockName= input("Ingrese el nombre de la empresa o activo para análisis de noticias (por ejemplo, Apple, Microsoft, Google): ").strip()
period = "max"
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

print(f"\n🚀 Análisis de {stockSymbol} ({stockName})")
print("=" * 50)

# Limpiar cache de noticias para nuevo análisis
from news_analysis import clear_sentiment_cache
clear_sentiment_cache()

# Obtener y procesar datos
stockData, predictors = setDataForTraining(stockSymbol, stockName)
print(f"📊 Datos: {len(stockData)} registros, {len(predictors)} predictores")

# Ejecutar backtesting
predictions = backtest(stockData, model, predictors, start=2500, step=250, stockSymbol=stockSymbol, stockName=stockName)

if not predictions.empty:
    # Métricas
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    accuracy = (predictions["Target"] == predictions["Predictions"]).mean()
    
    print(f"\n📈 Resultados:")
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    
    # Top predictores
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    #head de 5 predictores
    print(f"\n🏆 Top 5 predictores:")
    print(feature_importance.head(5).to_string(index=False))

else:
    print("❌ No se pudieron generar predicciones")

predictions.to_csv("predictions.csv")


