# Sistema de Predicción de Precios de Acciones 

## Tabla de Contenidos
1.Visión General de la Arquitectura del Sistema
2.Estructura y Flujo de Datos
3.Métodos de Captura de Datos
4.Análisis Estadístico
5.Tolerancia a Fallos
6.Comparación de Entornos de Ejecución
7.Conclusiones


## 1. Visión General de la Arquitectura del Sistema

El sistema de predicción de acciones está construido sobre una arquitectura distribuida basada en microservicios con los siguientes componentes clave:


```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐     ┌───────────────┐
│                 │     │               │     │                │     │               │
│  Finnhub API    │────▶│ Data Collector│────▶│  Kafka Broker  │────▶│ Model Trainer │
│                 │     │               │     │                │     │               │
└─────────────────┘     └───────────────┘     └────────────────┘     └───────┬───────┘
                                                                             │
                                                                             │
                                                                             ▼
                                                        ┌───────────────────────────────┐
                                                        │                               │
                                                        │  Streamlit Dashboard          │
                                                        │                               │
                                                        └───────────────────────────────┘
```


**Componentes principales:**
- **Recolector de Datos:** Obtiene datos en tiempo real de la API de Finnhub.
- **Corredor de Mensajes Kafka:** Maneja el flujo de datos entre componentes.
- **Entrenador de Modelos:** Procesa los datos y entrena modelos de aprendizaje automático.
- **Panel de Streamlit:** Visualiza datos y predicciones.
- **ZooKeeper:** Administra la configuración del clúster Kafka.

---

## 2. Estructura y Flujo de Datos

### Estructura de almacenamiento de datos:

```
/data
├── raw/                  # Raw stock data from API
├── processed/            # Processed data for model training
├── models/              # Trained model files
└── logs/                # Application logs
```


### Flujo de datos:
- El Recolector de Datos obtiene información de la API de Finnhub.
- Los datos crudos se:
  - Publican en tópicos de Kafka.
  - Guardan como archivos CSV para persistencia.
- El Entrenador de Modelos consume datos desde Kafka.
- Los datos procesados y modelos se almacenan en sus respectivas carpetas.
- El panel lee tanto datos crudos como procesados.

---

## 3. Métodos de Captura de Datos

### 1. Transmisión en tiempo real (API Streaming):
- Conexión directa a la API de Finnhub.
- Intervalos configurables (por defecto: 60 segundos).
- Captura precio actual, máximo, mínimo, apertura y cierre anterior.
- Implementa límites de tasa y manejo de errores.
- Retroceso exponencial ante fallos de la API.

### 2. Procesamiento por lotes:
- Recolección periódica de datos históricos.
- Almacenamiento como archivos CSV.
- Uso de Kafka para procesamiento en tiempo real.
- Validación y limpieza de datos.
- Partición automática por símbolo de acción.

---

## 4. Análisis Estadístico

### Estadísticas de Precio:
- Media móvil (ventana de 5 minutos)
- Desviación estándar móvil
- Precio medio ponderado por volumen (VWAP)
- Indicadores de momento

### Métricas del Modelo:
- Error Cuadrático Medio (MSE)
- R-cuadrado (R²)
- Precisión direccional
- Intervalos de confianza de la predicción

---

## 5. Tolerancia a Fallos

### Manejo de fallos de nodos secundarios:
- Chequeos de salud cada 30 segundos
- ZooKeeper supervisa el estado de los nodos
- Detección automática de fallos

### Proceso de recuperación:
- Reinicio automático del servicio (política `unless-stopped`)
- Replicación de datos entre nodos
- Persistencia de la cola de mensajes
- Rebalanceo automático de grupos de consumidores

### Consistencia de datos:
- Registros de transacciones en Kafka
- Checkpointing de datos
- Copia de seguridad en CSV
- Recuperación automática de datos

---

## 6. Comparación de Entornos de Ejecución

### Entorno Local

**Ventajas:**
- Iteraciones rápidas de desarrollo
- Depuración sencilla
- Bajos requerimientos de recursos
- Acceso directo al sistema de archivos

**Limitaciones:**
- Potencia limitada de procesamiento
- No hay alta disponibilidad
- Punto único de fallo

### Entorno Clúster

**Ventajas:**
- Escalabilidad horizontal
- Alta disponibilidad
- Balanceo de carga
- Tolerancia a fallos

**Limitaciones:**
- Configuración compleja
- Mayor sobrecarga operativa
- Latencia de red
- Coordinación de recursos necesaria

### Entorno Databricks

**Ventajas:**
- Infraestructura gestionada
- Monitoreo incorporado
- Escalado automático
- Integración con notebooks
- Rendimiento optimizado de Spark

**Limitaciones:**
- Dependencia de la plataforma
- Costo más alto
- Menor control sobre la infraestructura
- Requiere configuración adicional

---

## 7. Conclusiones

### Fortalezas de la Arquitectura:
- **Escalabilidad:** Arquitectura basada en microservicios que permite escalar componentes de forma independiente.
- **Confiabilidad:** Mecanismos de tolerancia a fallos aseguran estabilidad del sistema.
- **Flexibilidad:** Soporte para múltiples entornos de ejecución.
- **Procesamiento en Tiempo Real:** Canal eficiente de transmisión y procesamiento de datos.

### Diferencias Clave entre Entornos:

**Rendimiento:**
- Local: Ideal para desarrollo y pruebas.
- Clúster: Adecuado para cargas de trabajo en producción.
- Databricks: Mejor para procesamiento de datos a gran escala.

**Mantenimiento:**
- Local: Requiere poco mantenimiento.
- Clúster: Requiere equipo DevOps dedicado.
- Databricks: Servicio gestionado reduce la carga operativa.

**Eficiencia de Costos:**
- Local: Más económico para desarrollo.
- Clúster: Equilibrio entre costo y rendimiento.
- Databricks: Mayor costo pero menos gestión.

### Recomendaciones:
- Usar entorno local para desarrollo y pruebas.
- Desplegar en clúster para cargas en producción.
- Considerar Databricks para despliegues a gran escala o con servicio gestionado.
- Implementar monitoreo y alertas en todos los entornos.
- Mantener procedimientos de copia de seguridad y recuperación de datos.
