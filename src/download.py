"""Descarga CAMELS-COL desde Zenodo: solo los archivos que usa el proyecto.

Se omiten el raster de TWI (3,2 GB), la red de drenaje y las figuras.
Uso:  python src/download.py
"""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "18794895"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

ARCHIVOS = [
    "00_CAMELS-COL  Description.docx",
    "01_CAMELS_COL_Attributes.zip",
    "02_CAMELS_COL_Catchment_information.csv",
    "03_CAMELS_COL_Basin_boundary.zip",          # polígonos para el anidamiento
    "04_CAMELS_COL_Hydrometeorological_data.zip",
    "05_CAMELS_COL_Geologic_characteristics.csv",
    "06_CAMELS_COL_Land_cover_characteristics.csv",
    "07_CAMELS_COL_Soil_characteristics.csv",
    "08_CAMELS_COL_Climatic_indices.csv",
    "09_CAMELS_COL_Hydrological_signatures.csv",
    "10_CAMELS_COL_Physiograpic_characteristics.csv",
    "11_CAMELS_COL_Land_use_capability.csv",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API_URL) as r:
        registro = json.load(r)
    por_nombre = {f["key"]: f for f in registro["files"]}

    for nombre in ARCHIVOS:
        info = por_nombre[nombre]
        destino = RAW_DIR / nombre
        esperado = info["checksum"].split(":", 1)[1]

        if destino.exists() and md5(destino) == esperado:
            print(f"ya existe   {nombre}")
        else:
            print(f"descargando {nombre} ({info['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(info["links"]["self"], destino)
            if md5(destino) != esperado:
                raise RuntimeError(f"Checksum incorrecto en {nombre}")

        if destino.suffix == ".zip":
            carpeta = RAW_DIR / destino.stem
            if not carpeta.exists():
                with zipfile.ZipFile(destino) as z:
                    z.extractall(carpeta)
                print(f"  descomprimido en {carpeta.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
