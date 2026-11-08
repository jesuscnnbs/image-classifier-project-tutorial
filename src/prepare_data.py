"""Crea el arbol train/val/test que espera flow_from_directory.

El dataset original (`dogs-vs-cats/train`) es plano: 25.000 JPG con la clase
en el nombre del fichero (`cat.N.jpg`, `dog.N.jpg`). Este script hace un split
estratificado 80/10/10 y materializa las imagenes con *hardlinks*, de modo que
no se duplica espacio en disco ni se depende de rutas simbolicas.

Uso:
    python src/prepare_data.py
    python src/prepare_data.py --force   # regenera el split desde cero
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dogs-vs-cats" / "train"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_CSV = OUT_DIR / "split.csv"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def classify(path: Path) -> str | None:
    """Devuelve la clase a partir del nombre del fichero."""
    stem = path.stem.lower()
    if stem.startswith("cat"):
        return "cat"
    if stem.startswith("dog"):
        return "dog"
    return None


def collect_images() -> dict[str, list[Path]]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {RAW_DIR}")

    by_class: dict[str, list[Path]] = {"cat": [], "dog": []}
    skipped = 0
    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue
        label = classify(path)
        if label is None:
            skipped += 1
            continue
        by_class[label].append(path)

    if skipped:
        print(f"Aviso: {skipped} ficheros ignorados (sin etiqueta cat/dog).")
    for label, files in by_class.items():
        print(f"  {label}: {len(files)} imagenes")
    return by_class


def stratified_split(by_class: dict[str, list[Path]]) -> dict[str, list[tuple[Path, str]]]:
    """Reparte cada clase manteniendo la proporcion en los tres conjuntos."""
    rng = random.Random(SEED)
    result: dict[str, list[tuple[Path, str]]] = {name: [] for name in SPLITS}

    for label, files in by_class.items():
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        result["train"].extend((p, label) for p in files[:n_train])
        result["val"].extend((p, label) for p in files[n_train : n_train + n_val])
        result["test"].extend((p, label) for p in files[n_train + n_val :])

    for name, items in result.items():
        rng.shuffle(items)
        cats = sum(1 for _, label in items if label == "cat")
        print(f"  {name}: {len(items)} imagenes ({cats} cat / {len(items) - cats} dog)")
    return result


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink con fallback a copia si origen y destino estan en dispositivos distintos."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(splits: dict[str, list[tuple[Path, str]]]) -> None:
    for split, items in splits.items():
        for src, label in items:
            target_dir = OUT_DIR / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, target_dir / src.name)


def write_manifest(splits: dict[str, list[tuple[Path, str]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SPLIT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "split", "label"])
        for split, items in splits.items():
            for src, label in items:
                writer.writerow([src.name, split, label])
    print(f"Manifiesto escrito en {SPLIT_CSV.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borra data/processed y vuelve a generar el split.",
    )
    args = parser.parse_args()

    if args.force and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        print(f"Eliminado {OUT_DIR.relative_to(PROJECT_ROOT)}")

    print(f"Leyendo imagenes de {RAW_DIR.relative_to(PROJECT_ROOT)} ...")
    by_class = collect_images()
    print("Repartiendo (80/10/10, seed=42) ...")
    splits = stratified_split(by_class)

    print("Creando hardlinks en data/processed ...")
    materialize(splits)
    write_manifest(splits)
    print("Listo.")


if __name__ == "__main__":
    main()
