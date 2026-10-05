# Bank Deposit Subscription Predictor

Aplicación web con Streamlit que predice si un cliente contratará un depósito
bancario a plazo. El proyecto incluye el análisis exploratorio, el entrenamiento
del modelo y una interfaz para realizar predicciones.

## Estructura

```text
.
├── app/
│   └── streamlit_app.py          # Aplicación Streamlit
├── notebooks/
│   ├── Notebook_1.ipynb          # EDA y decisiones de evaluación
│   └── Notebook_2.ipynb          # Entrenamiento, evaluación y predicciones
├── data/
│   ├── bank_13.pkl               # Datos de entrenamiento
│   └── bank_competition.pkl      # Datos de competición
├── models/
│   └── modelo_final.joblib       # Pipeline y modelo entrenados
├── artifacts/
│   └── splits.pkl                # División serializada usada durante el trabajo
├── outputs/
│   └── predicciones.csv          # Predicciones generadas para competición
├── docs/
│   └── comprobacion_coincidencia.pdf
├── tests/
│   └── test_prediction.py
├── requirements.txt
└── README.md
```

Los datos, el modelo y los resultados se conservan para que la entrega
académica pueda reproducirse. Para ejecutar la aplicación se necesitan
`app/streamlit_app.py`, `models/modelo_final.joblib` y las dependencias
instaladas desde `requirements.txt`; los notebooks y los datasets no son
necesarios para hacer predicciones.

## Requisitos

- Python 3.10 o superior.
- La versión de `scikit-learn` usada para serializar el modelo: `1.8.0`.

El archivo `requirements.txt` incluye las dependencias de la aplicación, los
notebooks y las pruebas.

Instala las dependencias con:

```bash
python -m pip install -r requirements.txt
```

## Ejecutar la aplicación

Desde la raíz del repositorio:

```bash
python -m streamlit run app/streamlit_app.py
```

La aplicación estará disponible normalmente en
`http://localhost:8501`. Permite introducir los datos de un cliente, consultar
la probabilidad estimada y revisar dos casos de verificación incluidos en el
Notebook 2.

## Reproducir el análisis

1. Ejecuta `notebooks/Notebook_1.ipynb` para revisar el EDA y las decisiones de
   evaluación.
2. Ejecuta `notebooks/Notebook_2.ipynb` para entrenar y evaluar el modelo.
3. Comprueba que el modelo generado se guarda en `models/modelo_final.joblib`.
4. Ejecuta la aplicación siguiendo las instrucciones anteriores.

Los notebooks deben abrirse desde la raíz del repositorio para conservar las
rutas relativas a `data/`, `models/` y `outputs/`.

## Modelo y preprocesamiento

El pipeline utiliza un SVM con kernel RBF. Antes de la inferencia aplica
transformaciones personalizadas para:

- Crear `contacted_before` y normalizar el valor `-1` de `pdays`.
- Imputar valores ausentes de `marital`.
- Codificar y escalar las variables según el pipeline entrenado.

Las clases de los transformadores están definidas en
`app/streamlit_app.py` porque son necesarias para deserializar el modelo
serializado con `joblib`.

## Tests

Las pruebas automatizadas comprueban que el modelo carga correctamente y que
los dos casos de demostración mantienen las predicciones esperadas:

```bash
python -m pytest
```

La pestaña **Verificación del modelo** de la aplicación proporciona además una
comprobación visual de esos mismos casos.

## Limitaciones

El modelo es un artefacto entrenado para una práctica académica. Las
probabilidades no deben utilizarse como única base para decisiones financieras
reales. Los archivos `pickle` y `joblib` deben cargarse únicamente desde
fuentes confiables.
