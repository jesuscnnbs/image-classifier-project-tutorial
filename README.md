# Data Science Project Boilerplate

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## 🐶🐱 Image classification: dogs vs cats

Computer-vision project on the **Asirra** dataset (25,000 photos) using **EfficientNet-B0** and Keras 3.

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Stratified 80/10/10 split with hardlinks in `data/processed/{train,val,test}/{cat,dog}` + `split.csv` |
| `src/explore.ipynb` | EDA (9 dogs, 9 cats, sizes), generators, architecture and evaluation |
| `src/train.py` | Trains `--mode baseline` (`weights=None`) or `--mode transfer` (ImageNet + fine-tuning), with callbacks and evaluation |
| `src/app.py` | Inference (`--image`) or evaluation (`--evaluate`) |
| `src/backend.py` | Picks the Keras backend (PyTorch/MPS when available) |
| `models/` | `.keras` checkpoints, histories, metrics and `class_indices.json` |

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py
python src/train.py --mode baseline --tag baseline --epochs 2 --steps-per-epoch 250
python src/train.py --mode transfer --tag dogs_vs_cats_effnet --epochs 2 --fine-tune-epochs 1
python src/app.py --image dogs-vs-cats/train/dog.1.jpg
```

Key technical decisions:

- **224×224**, not 200×200: EfficientNet requires sizes that are multiples of 32.
- **No `rescale=1/255`**: EfficientNet normalizes inside the model and expects pixels in `[0, 255]`.
- With `shuffle=False`, `flow_from_directory` yields all `cat` samples first; validation therefore uses `shuffle=True` and test is evaluated in full.
- Callbacks monitor `val_accuracy` (in Keras 3 `val_acc` does not exist, so the checkpoint would never be saved).
- TensorFlow runs on CPU, but `pip install torch` enables MPS and roughly doubles throughput on Apple Silicon.

### Results

Evaluated on 2,500 held-out test images (Apple M1 CPU, PyTorch/MPS backend):

| Model | Training | Test accuracy | Test loss |
|---|---|---|---|
| EfficientNet-B0 from scratch (`weights=None`) | 2 epochs × 8,000 imgs | 63.0% | 0.669 |
| EfficientNet-B0 + ImageNet (frozen + fine-tuning) | 2 + 1 epochs × 8,000 imgs | **98.8%** | 0.032 |

The from-scratch baseline is capped at 2 epochs by CPU cost (≈30 min/epoch); the full `--epochs 10` run goes considerably higher. Transfer learning reaches 98.8% in ~17 min, well above the 80% reported by the 2007 SVM paper. Per-class metrics and confusion matrices are in `models/*_metrics.json`.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.