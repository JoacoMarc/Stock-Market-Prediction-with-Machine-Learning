import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from data_from_stock import getDataFromStock, setDataForTraining
from stock_analysis import backtest 
from news_analysis import clear_sentiment_cache
from stock_graph import create_graph
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 2)

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Acciones con ML")
        self.root.geometry("1200x800")  # Aumentado para acomodar gráficos
        
        # Variables para almacenar datos
        self.stock_data = None
        self.predictors = None
        self.predictions = None
        self.model = None
        self.feature_importance = None
        
        # Variable para controlar hilos
        self.analysis_thread = None
        self.analysis_running = False
        
        # Queue para comunicación thread-safe
        self.ui_queue = queue.Queue()
        
        # Inicializar modelo
        self.reset_model()
        
        self.setup_ui()
        
        # Iniciar el procesamiento de la queue
        self.process_ui_queue()
        
    def reset_model(self):
        """Resetea el modelo y limpia todos los datos"""
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        self.stock_data = None
        self.predictors = None
        self.predictions = None
        self.feature_importance = None
        
        # Limpiar figuras de matplotlib para evitar problemas de memoria
        try:
            plt.close('all')
        except:
            pass
    
    def process_ui_queue(self):
        """Procesa mensajes de la queue para actualizar UI de manera thread-safe"""
        try:
            while True:
                # Intentar obtener mensaje de la queue sin bloquear
                message = self.ui_queue.get_nowait()
                
                # Procesar diferentes tipos de mensajes
                if message['type'] == 'progress':
                    self.progress_var.set(message['value'])
                    if 'status' in message:
                        self.status_var.set(message['status'])
                        
                elif message['type'] == 'log':
                    self.logs_text.insert(tk.END, f"{message['text']}\n")
                    self.logs_text.see(tk.END)
                    
                elif message['type'] == 'metrics':
                    self.accuracy_var.set(message['accuracy'])
                    self.precision_var.set(message['precision'])
                    
                elif message['type'] == 'dataset_info':
                    self.dataset_info_var.set(message['info'])
                    
                elif message['type'] == 'predictors':
                    self.feature_importance = message['data']
                    self.update_predictors_display()
                    
                elif message['type'] == 'enable_graph':
                    self.update_graph()  # Actualizar gráfico automáticamente
                    
                elif message['type'] == 'analysis_complete':
                    self.analysis_running = False
                    self.analyze_button.config(state="normal")
                    self.cancel_button.config(state="disabled")
                    
                elif message['type'] == 'error':
                    messagebox.showerror("Error", message['text'])
                    self.analysis_running = False
                    self.analyze_button.config(state="normal")
                    self.cancel_button.config(state="disabled")
                    
        except queue.Empty:
            pass
        
        # Programar la próxima verificación de la queue
        self.root.after(100, self.process_ui_queue)
    
    def safe_update_progress(self, value, status=""):
        """Actualiza progreso de manera thread-safe"""
        self.ui_queue.put({
            'type': 'progress',
            'value': value,
            'status': status
        })
    
    def safe_log_message(self, message):
        """Añade mensaje al log de manera thread-safe"""
        self.ui_queue.put({
            'type': 'log',
            'text': message
        })
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar el grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Predicción de Acciones con Machine Learning", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame de entrada de datos
        input_frame = ttk.LabelFrame(main_frame, text="Datos de Entrada", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Símbolo de la acción
        ttk.Label(input_frame, text="Símbolo Bursátil:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.stock_symbol_var = tk.StringVar()
        self.stock_symbol_entry = ttk.Entry(input_frame, textvariable=self.stock_symbol_var, width=20)
        self.stock_symbol_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        ttk.Label(input_frame, text="(ej: AAPL, MSFT, GOOG)", foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Nombre de la empresa
        ttk.Label(input_frame, text="Nombre de la Empresa:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.stock_name_var = tk.StringVar()
        self.stock_name_entry = ttk.Entry(input_frame, textvariable=self.stock_name_var, width=20)
        self.stock_name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        ttk.Label(input_frame, text="(ej: Apple, Microsoft, Google)", foreground="gray").grid(row=1, column=2, sticky=tk.W, padx=(5, 0))
        
        # Frame para botones
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Botón de análisis
        self.analyze_button = ttk.Button(buttons_frame, text="🔍 Analizar Acción", 
                                        command=self.start_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón de cancelar (inicialmente oculto)
        self.cancel_button = ttk.Button(buttons_frame, text="❌ Cancelar Análisis", 
                                       command=self.cancel_analysis, state="disabled")
        self.cancel_button.pack(side=tk.LEFT)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(input_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Label de estado
        self.status_var = tk.StringVar(value="Listo para analizar")
        self.status_label = ttk.Label(input_frame, textvariable=self.status_var, foreground="white")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(2, weight=1)
        
        # Agregar callback para cuando se cambie de pestaña
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Pestaña de Métricas
        self.setup_metrics_tab()
        
        # Pestaña de Predictores
        self.setup_predictors_tab()
        
        # Pestaña de Gráficos
        self.setup_graphics_tab()
        
        # Pestaña de Logs
        self.setup_logs_tab()
        
    def setup_metrics_tab(self):
        """Configura la pestaña de métricas"""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="📈 Métricas")
        
        # Frame para métricas principales
        main_metrics_frame = ttk.LabelFrame(metrics_frame, text="Métricas de Rendimiento", padding="10")
        main_metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Accuracy
        accuracy_frame = ttk.Frame(main_metrics_frame)
        accuracy_frame.pack(fill=tk.X, pady=5)
        ttk.Label(accuracy_frame, text="Accuracy:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.accuracy_var = tk.StringVar(value="N/A")
        ttk.Label(accuracy_frame, textvariable=self.accuracy_var, font=("Arial", 12), 
                 foreground="green").pack(side=tk.LEFT, padx=(10, 0))
        
        # Precision
        precision_frame = ttk.Frame(main_metrics_frame)
        precision_frame.pack(fill=tk.X, pady=5)
        ttk.Label(precision_frame, text="Precision:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.precision_var = tk.StringVar(value="N/A")
        ttk.Label(precision_frame, textvariable=self.precision_var, font=("Arial", 12), 
                 foreground="green").pack(side=tk.LEFT, padx=(10, 0))
        
        # Información adicional
        info_frame = ttk.LabelFrame(metrics_frame, text="Información del Dataset", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.dataset_info_var = tk.StringVar(value="No hay datos cargados")
        ttk.Label(info_frame, textvariable=self.dataset_info_var, font=("Arial", 10)).pack()
        
    def setup_predictors_tab(self):
        """Configura la pestaña de predictores"""
        predictors_frame = ttk.Frame(self.notebook)
        self.notebook.add(predictors_frame, text="🏆 Predictores")
        
        # Frame para controles
        controls_frame = ttk.Frame(predictors_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Top Predictores por Importancia:", 
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        # Spinbox para seleccionar número de predictores
        ttk.Label(controls_frame, text="Mostrar:").pack(side=tk.RIGHT, padx=(0, 5))
        self.num_predictors_var = tk.StringVar(value="10")
        num_spinbox = ttk.Spinbox(controls_frame, from_=5, to=50, width=5, 
                                 textvariable=self.num_predictors_var,
                                 command=self.update_predictors_display)
        num_spinbox.pack(side=tk.RIGHT)
        
        # Treeview para mostrar predictores
        tree_frame = ttk.Frame(predictors_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.predictors_tree = ttk.Treeview(tree_frame, 
                                           columns=("Predictor", "Importancia"), 
                                           show="headings",
                                           yscrollcommand=tree_scroll_y.set,
                                           xscrollcommand=tree_scroll_x.set)
        
        # Configurar columnas
        self.predictors_tree.heading("Predictor", text="Predictor", anchor="center")
        self.predictors_tree.heading("Importancia", text="Importancia", anchor="center")
        
        # Configurar anchos y alineación de columnas
        self.predictors_tree.column("Predictor", width=300, anchor="center")
        self.predictors_tree.column("Importancia", width=150, anchor="center")
        
        self.predictors_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configurar scrollbars
        tree_scroll_y.config(command=self.predictors_tree.yview)
        tree_scroll_x.config(command=self.predictors_tree.xview)
        
    def setup_graphics_tab(self):
        """Configura la pestaña de gráficos"""
        graphics_frame = ttk.Frame(self.notebook)
        self.notebook.add(graphics_frame, text="📊 Gráficos")
        
        # Frame de controles
        controls_frame = ttk.LabelFrame(graphics_frame, text="Opciones de Gráfico", padding="10")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        controls_frame.columnconfigure(0, weight=1)  # Hacer que el frame se expanda
        
        # Frame interno para centrar los controles
        inner_frame = ttk.Frame(controls_frame)
        inner_frame.grid(row=0, column=0, sticky="")
        
        # Selector de tipo de gráfico
        ttk.Label(inner_frame, text="Tipo de Gráfico:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        self.graph_type_var = tk.StringVar(value="predicciones")
        graph_types = [
            ("Predicciones vs Real", "predicciones"),
            ("Accuracy por Período", "accuracy"),
            ("Distribución", "distribucion"),
            ("Importancia", "importancia")
        ]
        
        for i, (text, value) in enumerate(graph_types):
            ttk.Radiobutton(inner_frame, text=text, variable=self.graph_type_var, 
                           value=value, command=self.update_graph).grid(row=1, column=i, padx=15, pady=5)
        
        # Botón para actualizar gráfico
        update_button = ttk.Button(inner_frame, text="🔄 Actualizar Gráfico", 
                                  command=self.update_graph)
        update_button.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Frame para el gráfico
        self.graph_frame = ttk.LabelFrame(graphics_frame, text="Visualización", padding="5")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear figura matplotlib integrada con tamaño apropiado
        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        
        # Configurar el widget del canvas para que se expanda correctamente
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar para zoom, pan, etc.
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        
        # Gráfico inicial vacío
        self.create_empty_graph()
        
    def setup_logs_tab(self):
        """Configura la pestaña de logs"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs")
        
        # ScrolledText para mostrar logs
        self.logs_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, 
                                                  width=80, height=20)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botón para limpiar logs
        clear_button = ttk.Button(logs_frame, text="🗑️ Limpiar Logs", 
                                 command=self.clear_logs)
        clear_button.pack(pady=5)
        
    def log_message(self, message):
        """Añade un mensaje al log"""
        self.logs_text.insert(tk.END, f"{message}\n")
        self.logs_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_logs(self):
        """Limpia el área de logs"""
        self.logs_text.delete(1.0, tk.END)
        
    def on_tab_changed(self, event):
        """Callback para cuando se cambia de pestaña"""
        pass  # Simplificado para evitar problemas
    
    def refresh_graph(self):
        """Actualiza el gráfico para corregir problemas de layout"""
        pass  # Simplificado para evitar problemas
        
    def validate_inputs(self):
        """Valida las entradas del usuario"""
        stock_symbol = self.stock_symbol_var.get().strip().upper()
        stock_name = self.stock_name_var.get().strip()
        
        if not stock_symbol:
            messagebox.showerror("Error de Validación", "Por favor ingrese un símbolo bursátil")
            return False
            
        if len(stock_symbol) < 1 or len(stock_symbol) > 10:
            messagebox.showerror("Error de Validación", 
                               "El símbolo bursátil debe tener entre 1 y 10 caracteres")
            return False
            
        if not stock_name:
            messagebox.showwarning("Advertencia", 
                                 "Se recomienda ingresar el nombre de la empresa para mejor análisis de noticias")
            
        return True
        
    def start_analysis(self):
        """Inicia el análisis en un hilo separado"""
        if not self.validate_inputs():
            return
            
        # Verificar si ya hay un análisis en curso
        if self.analysis_running:
            messagebox.showwarning("Análisis en Curso", "Ya hay un análisis en progreso. Por favor espere a que termine.")
            return
            
        # Deshabilitar botón de análisis y habilitar cancelar
        self.analyze_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.clear_previous_results()
        
        # Resetear modelo y limpiar cache
        self.reset_model()
        clear_sentiment_cache()
        
        # Iniciar análisis en hilo separado
        self.analysis_thread = threading.Thread(target=self.perform_analysis)
        self.analysis_thread.daemon = True
        self.analysis_running = True
        self.analysis_thread.start()
        
    def cancel_analysis(self):
        """Cancela el análisis en curso"""
        if self.analysis_running:
            self.analysis_running = False
            self.safe_log_message("🛑 Cancelando análisis...")
            self.safe_update_progress(0, "Análisis cancelado")
            
            # Rehabilitar botones
            self.analyze_button.config(state="normal")
            self.cancel_button.config(state="disabled")
            
            messagebox.showinfo("Análisis Cancelado", "El análisis ha sido cancelado exitosamente.")
        
    def clear_previous_results(self):
        """Limpia los resultados de análisis anteriores"""
        self.accuracy_var.set("N/A")
        self.precision_var.set("N/A")
        self.dataset_info_var.set("Analizando...")
        
        # Limpiar tabla de predictores
        for item in self.predictors_tree.get_children():
            self.predictors_tree.delete(item)
            
        # Resetear barra de progreso
        self.progress_var.set(0)
        self.status_var.set("Preparando análisis...")
        
        # Limpiar variables de datos
        self.stock_data = None
        self.predictors = None
        self.predictions = None
        self.feature_importance = None
        
        # Crear gráfico vacío
        self.create_empty_graph()
        
    def perform_analysis(self):
        """Realiza el análisis completo"""
        try:
            stock_symbol = self.stock_symbol_var.get().strip().upper()
            stock_name = self.stock_name_var.get().strip()
            
            self.safe_update_progress(0, "Iniciando análisis...")
            self.safe_log_message(f"🚀 Iniciando análisis de {stock_symbol} ({stock_name})")
            
            # Limpiar cache de noticias adicional
            self.safe_update_progress(10, "Limpiando cache de noticias...")
            clear_sentiment_cache()
            self.safe_log_message("🧹 Cache de noticias limpiado")
            
            # Pequeña pausa para asegurar que el cache se limpie
            time.sleep(0.5)
            
            # Obtener y procesar datos
            self.safe_update_progress(20, "Obteniendo datos históricos...")
            self.safe_log_message("📊 Obteniendo datos históricos de la acción...")
            
            try:
                self.stock_data, self.predictors = setDataForTraining(stock_symbol, stock_name)
                self.safe_log_message(f"✅ Datos obtenidos: {len(self.stock_data)} registros, {len(self.predictors)} predictores")
                
                # Solo mostrar info básica inicialmente, las fechas reales se mostrarán después del backtesting
                dataset_info = f"Registros: {len(self.stock_data)} | Predictores: {len(self.predictors)} | Preparando análisis..."
                self.ui_queue.put({
                    'type': 'dataset_info',
                    'info': dataset_info
                })
                
            except Exception as e:
                raise Exception(f"Error al obtener datos: {str(e)}")
            
            # Verificar que el análisis no fue cancelado
            if not self.analysis_running:
                return
                
            # Ejecutar backtesting
            self.safe_update_progress(40, "Ejecutando backtesting...")
            self.safe_log_message("🔄 Ejecutando backtesting con modelo Random Forest...")
            
            try:
                self.predictions = backtest(self.stock_data, self.model, self.predictors, 
                                          start=2500, step=250, 
                                          stockSymbol=stock_symbol, stockName=stock_name)
                
                if self.predictions.empty:
                    raise Exception("No se pudieron generar predicciones")
                    
                self.safe_log_message(f"✅ Backtesting completado: {len(self.predictions)} predicciones generadas")
                
                # Actualizar información del dataset con fechas reales de predicciones
                start_date = self.predictions.index[0].strftime('%Y-%m-%d')
                end_date = self.predictions.index[-1].strftime('%Y-%m-%d')
                total_records = len(self.stock_data)
                prediction_records = len(self.predictions)
                
                dataset_info = f"Registros: {total_records} | Predictores: {len(self.predictors)} | Predicciones: {prediction_records} | Período: {start_date} a {end_date}"
                self.ui_queue.put({
                    'type': 'dataset_info',
                    'info': dataset_info
                })
                
            except Exception as e:
                raise Exception(f"Error en backtesting: {str(e)}")
            
            # Verificar que el análisis no fue cancelado
            if not self.analysis_running:
                return
                
            # Calcular métricas
            self.safe_update_progress(70, "Calculando métricas...")
            self.safe_log_message("📈 Calculando métricas de rendimiento...")
            
            try:
                precision = precision_score(self.predictions["Target"], self.predictions["Predictions"])
                accuracy = (self.predictions["Target"] == self.predictions["Predictions"]).mean()
                
                # Actualizar UI con métricas thread-safe
                self.ui_queue.put({
                    'type': 'metrics',
                    'accuracy': f"{accuracy:.4f} ({accuracy*100:.2f}%)",
                    'precision': f"{precision:.4f} ({precision*100:.2f}%)"
                })
                
                self.safe_log_message(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                self.safe_log_message(f"📊 Precision: {precision:.4f} ({precision*100:.2f}%)")
                
            except Exception as e:
                raise Exception(f"Error al calcular métricas: {str(e)}")
            
            # Verificar que el análisis no fue cancelado
            if not self.analysis_running:
                return
                
            # Calcular importancia de características
            self.safe_update_progress(85, "Calculando importancia de predictores...")
            self.safe_log_message("🏆 Calculando importancia de predictores...")
            
            try:
                feature_importance = pd.DataFrame({
                    'Feature': self.predictors,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Enviar datos de predictores thread-safe
                self.ui_queue.put({
                    'type': 'predictors',
                    'data': feature_importance
                })
                self.safe_log_message(f"✅ Top 5 predictores calculados")
                
            except Exception as e:
                raise Exception(f"Error al calcular importancia: {str(e)}")
            
            # Guardar predicciones
            self.safe_update_progress(95, "Guardando resultados...")
            try:
                self.predictions.to_csv("predictions.csv")
                self.safe_log_message("💾 Predicciones guardadas en predictions.csv")
            except Exception as e:
                self.safe_log_message(f"⚠️ Error al guardar: {str(e)}")
            
            # Completar
            if self.analysis_running:  # Solo completar si no fue cancelado
                self.safe_update_progress(100, "¡Análisis completado exitosamente!")
                self.safe_log_message("🎉 ¡Análisis completado exitosamente!")
                
                # Habilitar gráficos thread-safe
                self.ui_queue.put({'type': 'enable_graph'})
            
        except Exception as e:
            if self.analysis_running:  # Solo mostrar error si no fue cancelado
                self.safe_log_message(f"❌ Error: {str(e)}")
                self.safe_update_progress(0, f"Error: {str(e)}")
                self.ui_queue.put({
                    'type': 'error',
                    'text': f"Error durante el análisis:\n{str(e)}"
                })
            
        finally:
            # Notificar finalización thread-safe
            self.ui_queue.put({'type': 'analysis_complete'})
            
    def update_predictors_display(self):
        """Actualiza la visualización de predictores"""
        if self.feature_importance is None:
            return
            
        # Limpiar tabla actual
        for item in self.predictors_tree.get_children():
            self.predictors_tree.delete(item)
            
        # Obtener número de predictores a mostrar
        try:
            num_predictors = int(self.num_predictors_var.get())
        except:
            num_predictors = 10
            
        # Añadir predictores a la tabla
        top_predictors = self.feature_importance.head(num_predictors)
        for idx, row in top_predictors.iterrows():
            self.predictors_tree.insert("", tk.END, values=(row['Feature'], f"{row['Importance']:.6f}"))
    
    def create_empty_graph(self):
        """Crea un gráfico vacío inicial"""
        self.fig.clear()
        
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Realiza un análisis para ver gráficos\n📊', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14, color='gray',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
        
        # Configurar ejes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remover bordes
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Ajustar layout con márgenes apropiados
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()
    
    def update_graph(self):
        """Actualiza el gráfico según el tipo seleccionado"""
        if self.predictions is None or self.predictions.empty:
            self.create_empty_graph()
            return
            
        graph_type = self.graph_type_var.get()
        
        try:
            self.fig.clear()
            
            if graph_type == "predicciones":
                self.create_predictions_graph()
            elif graph_type == "accuracy":
                self.create_accuracy_graph()
            elif graph_type == "distribucion":
                self.create_distribution_graph()
            elif graph_type == "importancia":
                self.create_importance_graph()
                
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ Error creando gráfico: {str(e)}")
    
    def create_predictions_graph(self):
        """Crea gráfico de predicciones vs realidad"""
        ax = self.fig.add_subplot(111)
        
        # Usar todas las predicciones, no solo una muestra
        dates = self.predictions.index
        targets = self.predictions['Target']
        predictions = self.predictions['Predictions']
        
        # Crear gráfico de líneas más claro
        ax.plot(dates, targets, 'o-', color='blue', alpha=0.7, markersize=1.5, linewidth=1, label='Real')
        ax.plot(dates, predictions, 's-', color='red', alpha=0.7, markersize=1.5, linewidth=1, label='Predicción')
        
        ax.set_title(f'Predicciones vs Realidad ({len(self.predictions)} predicciones)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=10)
        ax.set_ylabel('Dirección (0=Baja, 1=Sube)', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Mejor formato de fechas
        import matplotlib.dates as mdates
        if len(dates) > 1000:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif len(dates) > 300:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Agregar estadísticas
        accuracy = (targets == predictions).mean()
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top', fontsize=9)
        
        # Ajustar layout para evitar recorte con padding adecuado
        self.fig.tight_layout(pad=2.0)
    
    def create_accuracy_graph(self):
        """Crea gráfico de accuracy por período"""
        ax = self.fig.add_subplot(111)
        
        # Calcular accuracy por ventanas de tiempo
        window_size = 100
        accuracies = []
        dates = []
        
        for i in range(window_size, len(self.predictions), 25):
            window_data = self.predictions.iloc[i-window_size:i]
            accuracy = (window_data['Target'] == window_data['Predictions']).mean()
            accuracies.append(accuracy)
            dates.append(window_data.index[-1])
        
        ax.plot(dates, accuracies, linewidth=2, color='blue', marker='o', markersize=3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Línea base (50%)')
        
        ax.set_title('Evolución del Accuracy por Período', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Mejor formato de fechas
        import matplotlib.dates as mdates
        if len(dates) > 50:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Ajustar layout para evitar recorte
        self.fig.tight_layout(pad=2.0)
    
    def create_distribution_graph(self):
        """Crea gráfico de distribución de predicciones"""
        ax = self.fig.add_subplot(111)
        
        # Contar predicciones
        pred_counts = self.predictions['Predictions'].value_counts()
        target_counts = self.predictions['Target'].value_counts()
        
        x = ['Baja (0)', 'Sube (1)']
        pred_values = [pred_counts.get(0, 0), pred_counts.get(1, 0)]
        target_values = [target_counts.get(0, 0), target_counts.get(1, 0)]
        
        x_pos = range(len(x))
        width = 0.35
        
        ax.bar([p - width/2 for p in x_pos], pred_values, width, label='Predicciones', color='orange', alpha=0.7)
        ax.bar([p + width/2 for p in x_pos], target_values, width, label='Realidad', color='blue', alpha=0.7)
        
        ax.set_title('Distribución de Predicciones vs Realidad', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dirección del Mercado', fontsize=10)
        ax.set_ylabel('Cantidad de Días', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for i, (pred, target) in enumerate(zip(pred_values, target_values)):
            ax.text(i - width/2, pred + pred*0.01, str(pred), ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, target + target*0.01, str(target), ha='center', va='bottom', fontsize=8)
            
        # Ajustar layout para evitar recorte
        self.fig.tight_layout(pad=2.0)
    
    def create_importance_graph(self):
        """Crea gráfico de importancia de predictores"""
        if self.feature_importance is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay datos de importancia disponibles', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12, color='gray')
            return
            
        ax = self.fig.add_subplot(111)
        
        # Top 15 predictores
        top_features = self.feature_importance.head(15)
        
        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features['Importance'], color='skyblue', alpha=0.7)
        
        ax.set_title('Top 15 Predictores por Importancia', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importancia', fontsize=10)
        ax.set_ylabel('Predictores', fontsize=10)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Invertir el eje y para mostrar el más importante arriba
        ax.invert_yaxis()
        
        # Ajustar layout para evitar recorte
        self.fig.tight_layout(pad=2.0)
            
    def generate_graph(self):
        """Genera el gráfico de predicciones (método legacy para compatibilidad)"""
        self.update_graph()

def main():
    """Función principal"""
    root = tk.Tk()
    app = StockPredictionGUI(root)
    
    def on_closing():
        """Función para limpiar recursos al cerrar"""
        try:
            # Cancelar análisis en curso si existe
            if hasattr(app, 'analysis_running') and app.analysis_running:
                app.analysis_running = False
            
            # Limpiar figuras de matplotlib
            plt.close('all')
            
            # Limpiar cache de noticias
            clear_sentiment_cache()
            
        except Exception as e:
            print(f"Error durante limpieza: {e}")
        finally:
            root.destroy()
    
    # Configurar el protocolo de cierre
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
        on_closing()
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        on_closing()

if __name__ == "__main__":
    main()