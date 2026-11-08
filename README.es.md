# Plantilla de Proyecto de Ciencia de Datos

Esta plantilla está diseñada para impulsar proyectos de ciencia de datos proporcionando una configuración básica para conexiones de base de datos, procesamiento de datos, y desarrollo de modelos de aprendizaje automático. Incluye una organización estructurada de carpetas para tus conjuntos de datos y un conjunto de paquetes de Python predefinidos necesarios para la mayoría de las tareas de ciencia de datos.

## 🐶🐱 Clasificador de imágenes: perros vs gatos

Proyecto de visión por computador sobre el dataset **Asirra** (25.000 fotos) con **EfficientNet-B0** y Keras 3.

| Fichero | Qué hace |
|---|---|
| `src/prepare_data.py` | Split estratificado 80/10/10 con hardlinks en `data/processed/{train,val,test}/{cat,dog}` + `split.csv` |
| `src/explore.ipynb` | EDA (9 perros, 9 gatos, tamaños), generadores, arquitectura y evaluación |
| `src/train.py` | Entrena `--mode baseline` (`weights=None`) o `--mode transfer` (ImageNet + fine-tuning), con callbacks y evaluación |
| `src/app.py` | Inferencia (`--image`) o evaluación (`--evaluate`) |
| `src/backend.py` | Selecciona el backend de Keras (PyTorch/MPS si está instalado) |
| `models/` | Checkpoints `.keras`, historiales, métricas y `class_indices.json` |

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py
python src/train.py --mode baseline --tag baseline --epochs 2 --steps-per-epoch 250
python src/train.py --mode transfer --tag dogs_vs_cats_effnet --epochs 2 --fine-tune-epochs 1
python src/app.py --image dogs-vs-cats/train/dog.1.jpg
```

Decisiones técnicas relevantes:

- **224×224**, no 200×200: EfficientNet exige lados múltiplos de 32.
- **Sin `rescale=1/255`**: EfficientNet normaliza dentro del modelo y espera píxeles en `[0, 255]`.
- Con `shuffle=False`, `flow_from_directory` sirve primero todas las `cat`; por eso la validación usa `shuffle=True` y el test se evalúa completo.
- El monitor de los callbacks es `val_accuracy` (en Keras 3 `val_acc` no existe y el checkpoint nunca se guardaría).
- TensorFlow funciona en CPU, pero `pip install torch` activa MPS y duplica la velocidad en Apple Silicon.

### Resultados

Evaluación sobre 2.500 imágenes de test no vistas durante el entrenamiento (CPU M1, backend PyTorch/MPS):

| Modelo | Entrenamiento | Test accuracy | Test loss |
|---|---|---|---|
| EfficientNet-B0 desde cero (`weights=None`) | 2 épocas × 8.000 imgs | 63,0% | 0,669 |
| EfficientNet-B0 + ImageNet (congelado + fine-tuning) | 2 + 1 épocas × 8.000 imgs | **98,8%** | 0,032 |

El baseline supera el 50% de azar pero está lejos de su techo: se queda en 2 épocas por el coste de entrenar desde cero en CPU (≈30 min/época). Con el run completo (`--epochs 10`) sube bastante. El transfer learning alcanza el 98,8% en ~17 min y supera ampliamente el 80% del SVM de 2007. Detalle por clase y matriz de confusión en `models/*_metrics.json`.

## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## ⚡ Configuración Inicial en Codespaces (Recomendado)

No es necesario realizar ninguna configuración manual, ya que **Codespaces se configura automáticamente** con los archivos predefinidos que ha creado la academia para ti. Simplemente sigue estos pasos:

1. **Espera a que el entorno se configure automáticamente**.
   - Todos los paquetes necesarios y la base de datos se instalarán por sí mismos.
   - El `username` y `db_name` creados automáticamente están en el archivo **`.env`** en la raíz del proyecto.
2. **Una vez que Codespaces esté listo, puedes comenzar a trabajar inmediatamente**.


## 💻 Configuración en Local (Solo si no puedes usar Codespaces)

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: 

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER mi_usuario WITH PASSWORD 'mi_contraseña'; 
    CREATE DATABASE mi_base_de_datos OWNER mi_usuario; 
END \$\$;"
```
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: 

```bash
$ psql -U mi_usuario -d mi_base_de_datos
```

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="postgresql://<USUARIO>:<CONTRASEÑA>@<HOST>:<PUERTO>/<NOMBRE_BD>"

#example
DATABASE_URL="postgresql://mi_usuario:mi_contraseña@localhost:5432/mi_base_de_datos"
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
python src/app.py
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.