"""Entrena el clasificador perro/gato con EfficientNet-B0.

Dos modos:
  * baseline: backbone con pesos aleatorios (`weights=None`), como en el enunciado.
  * transfer: backbone preentrenado en ImageNet, congelado y luego ajustado.

Uso:
    python src/train.py --mode baseline --tag baseline --epochs 10
    python src/train.py --mode transfer --tag dogs_vs_cats_effnet --fine-tune-epochs 5
    python src/train.py --mode baseline --tag smoke --smoke   # valida el pipeline

El mejor checkpoint se guarda en models/<tag>.keras y las metricas de test en
models/<tag>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend import configure_backend

BACKEND = configure_backend()  # debe ejecutarse antes de importar keras

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import callbacks, layers  # noqa: E402
from keras.applications import EfficientNetB0  # noqa: E402

try:  # Keras <= 3.10
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:  # Keras 3.15+ lo movio al arbol legacy
    try:
        from keras.src.legacy.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMG_SIZE = (224, 224)
SEED = 42


def build_generators(batch_size: int):
    """Generadores de train/val/test. Sin `rescale`: EfficientNet ya normaliza."""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator()

    common = dict(
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
    )
    trdata = train_datagen.flow_from_directory(
        DATA_DIR / "train", subset=None, shuffle=True, **common
    )
    # shuffle=True: con shuffle=False el generador sirve primero todas las `cat`,
    # asi que un `validation_steps` corto mediria solo una clase.
    valdata = eval_datagen.flow_from_directory(DATA_DIR / "val", shuffle=True, **common)
    # test debe ir sin shuffle para que `tsdata.classes` siga alineado con las predicciones.
    tsdata = eval_datagen.flow_from_directory(DATA_DIR / "test", shuffle=False, **common)
    return trdata, valdata, tsdata


def build_model(mode: str) -> keras.Model:
    weights = "imagenet" if mode == "transfer" else None
    base = EfficientNetB0(include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3))
    # Desde cero hay que entrenar el backbone; con pesos de ImageNet empezamos congelandolo.
    base.trainable = weights is None

    model = keras.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=2, activation="softmax"),
        ],
        name=f"efficientnetb0_{mode}",
    )
    return model


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(
    tag: str, initial_value_threshold: float | None = None
) -> list[callbacks.Callback]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{tag}.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            # Evita que la fase de fine-tuning pise un checkpoint mejor de la fase anterior.
            initial_value_threshold=initial_value_threshold,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def append_history(history: dict, other: keras.callbacks.History | dict) -> None:
    """Fusiona un `History` de Keras o un dict de metricas ya normalizado."""
    entries = other.history if isinstance(other, keras.callbacks.History) else other
    for key, values in entries.items():
        history.setdefault(key, []).extend(float(v) for v in values)


def train(
    model: keras.Model,
    trdata,
    valdata,
    callbacks_list: list[callbacks.Callback],
    epochs: int,
    steps_per_epoch: int | None,
    validation_steps: int | None,
) -> dict:
    history: dict[str, list[float]] = {}

    started = time.time()
    hist = model.fit(
        trdata,
        epochs=epochs,
        validation_data=valdata,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=2,
    )
    elapsed = time.time() - started
    append_history(history, hist)
    print(f"Entrenamiento terminado en {elapsed / 60:.1f} min")

    if "val_accuracy" in history:
        best = max(history["val_accuracy"])
        print(f"Mejor val_accuracy: {best:.4f}")
    return history


def evaluate(tag: str, tsdata) -> dict:
    """Evalua el mejor checkpoint sobre test y guarda las metricas."""
    from sklearn.metrics import classification_report, confusion_matrix

    model = keras.models.load_model(MODELS_DIR / f"{tag}.keras")
    loss, acc = model.evaluate(tsdata, verbose=0)
    print(f"\nTest loss={loss:.4f}  accuracy={acc:.4f}")

    tsdata.reset()
    probs = model.predict(tsdata, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = tsdata.classes

    class_indices = tsdata.class_indices
    names = [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]
    report = classification_report(
        y_true, y_pred, target_names=names, digits=4, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred)
    print(report)
    print("Matriz de confusion (filas=real, columnas=predicho):")
    print(matrix)

    metrics = {
        "tag": tag,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
        "class_indices": class_indices,
        "n_test": int(tsdata.samples),
    }
    with (MODELS_DIR / f"{tag}_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with (MODELS_DIR / "class_indices.json").open("w", encoding="utf-8") as fh:
        json.dump(class_indices, fh, indent=2)
    return metrics


def save_history(tag: str, history: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{tag}_history.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Historial guardado en {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "transfer"], default="baseline")
    parser.add_argument("--tag", default=None, help="Nombre de los artefactos en models/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run corto para validar el pipeline")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evalua un checkpoint ya entrenado (models/<tag>.keras) sin reentrenar.",
    )
    args = parser.parse_args()

    tag = args.tag or args.mode
    if args.smoke:
        args.epochs = 2
        args.steps_per_epoch = args.steps_per_epoch or 50
        args.validation_steps = args.validation_steps or 20
        tag = args.tag or f"{args.mode}_smoke"

    keras.utils.set_random_seed(SEED)
    print(f"Backend Keras: {BACKEND}")

    if not (DATA_DIR / "train").is_dir():
        raise SystemExit(
            "No existe data/processed/train. Ejecuta primero: python src/prepare_data.py"
        )

    trdata, valdata, tsdata = build_generators(args.batch_size)
    print(f"Clases: {trdata.class_indices}")
    print(f"Train: {trdata.samples} | Val: {valdata.samples} | Test: {tsdata.samples}")

    if args.evaluate_only:
        evaluate(tag, tsdata)
        return

    model = build_model(args.mode)
    compile_model(model, args.lr)
    model.summary(print_fn=lambda line: None)  # silencioso; evita ruido en logs

    history = train(
        model,
        trdata,
        valdata,
        make_callbacks(tag),
        args.epochs,
        args.steps_per_epoch,
        args.validation_steps,
    )
    save_history(tag, history)

    if args.mode == "transfer" and args.fine_tune_epochs > 0:
        for layer in model.layers[0].layers[-30:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        compile_model(model, args.fine_tune_lr)
        best_so_far = max(history.get("val_accuracy", [0.0]))
        print(f"\nFine-tuning de los ultimos 30 bloques (lr={args.fine_tune_lr})")
        hist = train(
            model,
            trdata,
            valdata,
            make_callbacks(tag, initial_value_threshold=best_so_far),
            args.fine_tune_epochs,
            args.steps_per_epoch,
            args.validation_steps,
        )
        append_history(history, hist)
        save_history(tag, history)

    if not args.no_evaluate:
        evaluate(tag, tsdata)


if __name__ == "__main__":
    main()
