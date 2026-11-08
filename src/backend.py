"""Selecciona el backend de Keras antes de importarlo.

En Apple Silicon, TensorFlow solo usa CPU y EfficientNet-B0 (convoluciones
depthwise) resulta muy lento. Keras 3 puede usar PyTorch con la GPU via MPS,
que es entre 5 y 15 veces mas rapido. Si `KERAS_BACKEND` ya esta definido, se
respeta; si no, se usa PyTorch cuando esta instalado y TensorFlow en caso
contrario.
"""

from __future__ import annotations

import importlib.util
import os


def configure_backend() -> str:
    if "KERAS_BACKEND" in os.environ:
        return os.environ["KERAS_BACKEND"]
    os.environ["KERAS_BACKEND"] = (
        "torch" if importlib.util.find_spec("torch") is not None else "tensorflow"
    )
    return os.environ["KERAS_BACKEND"]
