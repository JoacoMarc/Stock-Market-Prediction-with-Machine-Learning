import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from news_analysis import sentiment_analysis


def predict_with_sentiment(train, test, predictors, model, stockSymbol, stockName, sentiment_stats=None):
    # Entrenar el modelo
    model.fit(train[predictors], train["Target"])

    # Para datos de test recientes, intentar actualizar sentiment
    test_with_updated_sentiment = test.copy()

    # Obtener fecha actual real (sin zona horaria)
    current_date = pd.Timestamp.now().tz_localize(None)
    
    # Calcular rango de fechas donde las noticias están disponibles
    # La API de noticias tiene datos disponibles aproximadamente del último mes
    earliest_news_date = current_date - pd.Timedelta(days=30)
    latest_news_date = current_date - pd.Timedelta(days=1)  # Hasta ayer
    
    # Para cada fecha de test, determinar si podemos usar sentiment
    for test_date in test.index:
        # Normalizar test_date quitando zona horaria si la tiene
        test_date_normalized = test_date.tz_localize(None) if test_date.tz is not None else test_date
        
        # Solo buscar noticias si:
        # 1. La fecha de predicción está en el rango donde tenemos noticias disponibles
        # 2. No estamos prediciendo el futuro
        if (earliest_news_date <= test_date_normalized <= latest_news_date):
            try:
                # CLAVE: Solo usar noticias hasta el día ANTERIOR al que estamos prediciendo
                max_news_date = (test_date_normalized - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Verificar que la fecha de noticias no sea futura
                max_news_date_dt = pd.to_datetime(max_news_date).tz_localize(None)
                if max_news_date_dt <= latest_news_date:
                    pos, neg, neu, _ = sentiment_analysis(stockSymbol, stockName, max_date=max_news_date)

                    if pos > 0:
                        test_with_updated_sentiment.loc[test_date, "Sentiment_Positive"] = pos
                        test_with_updated_sentiment.loc[test_date, "Sentiment_Negative"] = neg
                        test_with_updated_sentiment.loc[test_date, "Sentiment_Neutral"] = neu
                        
                        # ARREGLO: Actualizar estadísticas
                        if sentiment_stats:
                            sentiment_stats['sentiment_applied'] += 1
                else:
                    if sentiment_stats:
                        sentiment_stats['skipped_future'] += 1
                    
            except Exception as e:
                if sentiment_stats:
                    sentiment_stats['skipped_other'] += 1
        else:
            # Para fechas fuera del rango de noticias, usar valores neutros sin búsqueda
            if test_date_normalized < earliest_news_date:
                if sentiment_stats:
                    sentiment_stats['skipped_old'] += 1
            elif test_date_normalized > latest_news_date:
                if sentiment_stats:
                    sentiment_stats['skipped_future'] += 1
            else:
                if sentiment_stats:
                    sentiment_stats['skipped_other'] += 1
            
    
    # Asegurar que todas las columnas de predictors existen
    for col in predictors:
        if col not in test_with_updated_sentiment.columns:
            if 'Sentiment' in col:
                test_with_updated_sentiment[col] = 0.33  # Neutral value
            else:
                return pd.DataFrame()  # Return empty if critical columns missing
    
    try:
        preds = model.predict_proba(test_with_updated_sentiment[predictors])[:, 1]
        preds = [1 if x >= 0.5 else 0 for x in preds]
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined
    except Exception as e:
        print(f"Error making predictions: {e}")
        return pd.DataFrame()


def backtest(stockData, model, predictors, start=2500, step=250, stockSymbol=None, stockName=None):
    """
    Backtesting mejorado con sentiment cuando sea relevante
    """
    all_predictions = []
    
    # Estadísticas para el análisis de noticias
    sentiment_stats = {
        'total_predictions': 0,
        'sentiment_applied': 0,
        'skipped_old': 0,
        'skipped_future': 0,
        'skipped_other': 0
    }

    for i in range(start, stockData.shape[0], step):
        train = stockData.iloc[0:i].copy()
        test = stockData.iloc[i:(i + step)].copy()
        
        # Contar predicciones totales
        sentiment_stats['total_predictions'] += len(test)

        predictions = predict_with_sentiment(train, test, predictors, model, stockSymbol, stockName, sentiment_stats)
        if not predictions.empty:
            all_predictions.append(predictions)
    
    # Mostrar estadísticas al final
    print(f"\n📊 Estadísticas del análisis de noticias:")
    print(f"  Total de predicciones: {sentiment_stats['total_predictions']}")
    print(f"  Con sentiment aplicado: {sentiment_stats['sentiment_applied']}")
    print(f"  Saltadas (muy antiguas): {sentiment_stats['skipped_old']}")
    print(f"  Saltadas (futuras): {sentiment_stats['skipped_future']}")
    if sentiment_stats['total_predictions'] > 0:
        pct_with_news = (sentiment_stats['sentiment_applied'] / sentiment_stats['total_predictions']) * 100
        print(f"  Porcentaje con noticias: {pct_with_news:.1f}%")

    if all_predictions:
        return pd.concat(all_predictions)
    else:
        return pd.DataFrame()


