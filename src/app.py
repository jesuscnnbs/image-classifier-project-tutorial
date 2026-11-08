"""Punto de entrada del clasificador perro/gato.

Uso:
    python src/app.py --image ruta/a/foto.jpg
    python src/app.py --evaluate

Requiere el split generado (python src/prepare_data.py) y un modelo entrenado
(python src/train.py ...), guardado en models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from backend import configure_backend

BACKEND = configure_backend()  # debe ejecutarse antes de importar keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
IMG_SIZE = (224, 224)
DEFAULT_MODEL = MODELS_DIR / "dogs_vs_cats_effnet.keras"


def load_class_indices(model_path: Path) -> dict[str, int]:
    manifest = MODELS_DIR / "class_indices.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    # flow_from_directory ordena las clases alfabeticamente: cat=0, dog=1
    return {"cat": 0, "dog": 1}


def predict(model_path: Path, image_path: Path) -> dict:
    import numpy as np
    import keras

    model = keras.models.load_model(model_path)
    class_indices = load_class_indices(model_path)
    index_to_name = {index: name for name, index in class_indices.items()}

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        # EfficientNet de Keras normaliza dentro del modelo: pixels en [0, 255].
        batch = np.asarray(img, dtype="float32")[None, ...]

    probs = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probs))
    return {
        "image": str(image_path),
        "prediction": index_to_name.get(best, str(best)),
        "confidence": float(probs[best]),
        "probabilities": {
            index_to_name.get(i, str(i)): float(p) for i, p in enumerate(probs)
        },
    }


def evaluate_model(model_path: Path) -> dict:
    import keras

    try:
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report

    test_gen = ImageDataGenerator().flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    model = keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    report = classification_report(
        test_gen.classes,
        model.predict(test_gen, verbose=0).argmax(axis=1),
        target_names=list(test_gen.class_indices),
        digits=4,
    )
    print(f"Modelo: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Test loss={loss:.4f}  accuracy={accuracy:.4f}\n")
    print(report)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Imagen a clasificar")
    group.add_argument("--evaluate", action="store_true", help="Evalua el modelo sobre test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"No existe el modelo {args.model}. Entrena primero con: python src/train.py"
        )

    if args.evaluate:
        evaluate_model(args.model)
        return

    result = predict(args.model, args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
