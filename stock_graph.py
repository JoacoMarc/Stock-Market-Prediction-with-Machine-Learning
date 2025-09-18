
import matplotlib.pyplot as plt

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