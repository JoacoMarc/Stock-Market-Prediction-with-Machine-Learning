
import matplotlib.pyplot as plt
import pandas as pd

def create_graph(predictions_data, stock_symbol):
    """
    Crea un gráfico de las predicciones vs targets reales
    """
    plt.figure(figsize=(15, 8))
    
    # Crear subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Gráfico 1: Predicciones vs Target real
    predictions_data['Target'].plot(ax=ax1, label='Target Real', color='blue', alpha=0.7)
    predictions_data['Predictions'].plot(ax=ax1, label='Predicciones', color='red', alpha=0.7)
    ax1.set_title(f'Predicciones vs Target Real - {stock_symbol}')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor (0=Baja, 1=Sube)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Diferencia entre predicción y realidad
    diff = predictions_data['Predictions'] - predictions_data['Target']
    diff.plot(ax=ax2, color='purple', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Diferencia (Predicción - Target Real)')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Diferencia')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
