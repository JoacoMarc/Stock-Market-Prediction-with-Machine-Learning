import requests
from transformers import pipeline
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta

# Suprimir warnings y mensajes verbosos
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Cache global para evitar requests duplicadas
_sentiment_cache = {}
_bulk_news_cache = {}  # Cache para noticias en bulk

def get_news(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_bulk_news_for_period(stockSymbol, stockName=None, days_back=30):
    """
    Obtiene todas las noticias de los últimos N días en una sola llamada
    y las organiza por fecha para uso eficiente
    """
    global _bulk_news_cache
    
    cache_key = f"{stockSymbol}_{days_back}"
    
    
    # Si ya tenemos las noticias en cache, devolverlas
    if cache_key in _bulk_news_cache:
        return _bulk_news_cache[cache_key]

    current_date = pd.Timestamp.now()
    from_date = (current_date - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Términos de búsqueda más específicos para finanzas
    financial_terms = ["stock", "earnings", "revenue", "shares", "financial"]
    if stockName:
        search_term = f'("{stockSymbol}" OR "{stockName}") AND ({" OR ".join(financial_terms)})'
    else:
        search_term = f'"{stockSymbol}" AND ({" OR ".join(financial_terms)})'
    
    # Dominios financieros confiables
    financial_domains = "reuters.com,bloomberg.com,marketwatch.com,cnbc.com,finance.yahoo.com,wsj.com,ft.com,nasdaq.com"
    
    url = f"https://newsapi.org/v2/everything?q={search_term}&apiKey=74bd02775bbe4b4a801ee1e8b5dbc8dd&language=en&from={from_date}&to={to_date}&domains={financial_domains}&sortBy=relevancy&pageSize=100"
    
    print(f"🔄 Obteniendo noticias para {stockSymbol} de los últimos {days_back} días...")
    news = get_news(url)
    news_by_date = {}
    
    if news:
        print(f"📰 API devolvió {len(news.get('articles', []))} artículos")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            top_k=None  
        )
        
        for article in news.get("articles", []):
            try:
                published_date = article.get('publishedAt', '')
                if not published_date:
                    continue
                    
                # Extraer solo la fecha (sin hora)
                article_date = pd.to_datetime(published_date).strftime('%Y-%m-%d')
                
                source = article.get('source', {}).get('name')
                content = article.get('content') or article.get('description', '')
                title = article.get('title', '')

                # Verificar relevancia del artículo
                if content and len(content.strip()) > 10:
                    content_lower = content.lower()
                    title_lower = title.lower()
                    symbol_lower = stockSymbol.lower()
                    name_lower = stockName.lower() if stockName else ""
                    
                
                
                    # Analizar sentimiento
                    text_to_analyze = f"{title} {content}"[:512]
                    sentiment_results = sentiment_analyzer(text_to_analyze)
                    
                    main_prediction = max(sentiment_results[0], key=lambda x: x['score'])
                    
                    sentiment_data = {
                        'source': source,
                        'sentiment': main_prediction['label'],
                        'confidence': main_prediction['score'],
                        'positive_score': next((s['score'] for s in sentiment_results[0] if s['label'] == 'positive'), 0),
                        'negative_score': next((s['score'] for s in sentiment_results[0] if s['label'] == 'negative'), 0),
                        'neutral_score': next((s['score'] for s in sentiment_results[0] if s['label'] == 'neutral'), 0),
                        'title': title
                    }
                    
                    # Agrupar por fecha
                    if article_date not in news_by_date:
                        news_by_date[article_date] = []
                    news_by_date[article_date].append(sentiment_data)
                    
            except Exception as e:
                continue
        print(f"📅 Fechas con noticias: {list(news_by_date.keys())}")
    else:
        print("❌ La API no devolvió noticias")
    
    # Calcular sentiment promedio por fecha
    sentiment_by_date = {}
    for date, articles in news_by_date.items():
        if articles:
            df = pd.DataFrame(articles)
            positive = (df['positive_score'] * df['confidence']).sum() / df['confidence'].sum()
            negative = (df['negative_score'] * df['confidence']).sum() / df['confidence'].sum()
            neutral = (df['neutral_score'] * df['confidence']).sum() / df['confidence'].sum()
            sentiment_by_date[date] = (positive, negative, neutral, len(articles))
    
    print(f"✅ Procesadas {len(sentiment_by_date)} fechas con noticias")
    
    # Guardar en cache
    _bulk_news_cache[cache_key] = sentiment_by_date
    
    return sentiment_by_date


def sentiment_analysis(stockSymbol, stockName=None, max_date=None):
    """
    Función optimizada que usa el cache de noticias en bulk
    """
    global _sentiment_cache
    
    # Si no especifican fecha máxima, usar hasta ayer
    if max_date is None:
        max_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Verificar que la fecha no sea futura
    max_date_dt = pd.to_datetime(max_date)
    current_date = pd.Timestamp.now()
    
    if max_date_dt >= current_date:
        max_date = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Verificar si la fecha está en el rango disponible de la API (últimos 30 días)
    oldest_available = current_date - pd.Timedelta(days=30)
    
    # Si la fecha es muy antigua, devolver valores neutros
    if max_date_dt < oldest_available:
        return (0.33, 0.33, 0.34, None)
    
    # Crear clave única para el cache
    cache_key = f"{stockSymbol}_{max_date}"
    
    # Si ya tenemos el resultado, devolverlo
    if cache_key in _sentiment_cache:
        return _sentiment_cache[cache_key]
    
    # Obtener noticias en bulk (se hace una sola vez y se cachea)
    bulk_news = get_bulk_news_for_period(stockSymbol, stockName, days_back=30)
    
    # Buscar sentiment para la fecha específica
    if max_date in bulk_news:
        positive, negative, neutral, count = bulk_news[max_date]
        result = (positive, negative, neutral, max_date)
    else:
        # Si no hay noticias para esa fecha específica, usar valores neutros
        result = (0.33, 0.33, 0.34, None)
    
    # Guardar en cache
    _sentiment_cache[cache_key] = result
    
    return result

# Función para limpiar cache si es necesario
def clear_sentiment_cache():
    global _sentiment_cache, _bulk_news_cache
    _sentiment_cache = {}
    _bulk_news_cache = {}
