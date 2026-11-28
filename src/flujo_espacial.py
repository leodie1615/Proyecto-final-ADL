"""Etapa 2 (flujo espacial, cuenca no vista): entrenamiento y evaluación de las 5 particiones.

Uso, desde la raíz del repositorio:
    python src/flujo_espacial.py entrenar --semillas 42 43 44
    python src/flujo_espacial.py estado
    python src/flujo_espacial.py entrenar --variante era5l --semillas 42 43 44   # mejora B1 del plan
    python src/flujo_espacial.py entrenar --variante transformer --semillas 42 --sin-temporal   # B3

Variantes (`--variante`):
- `base`: forzantes de CAMELS-COL (`configs/base.yml`, `data/processed`). Corridas `espacial_p{k}_s{semilla}`.
- `era5l`: la configuración C2 de la etapa 1 (`configs/corridas/confirmacion_era5l_s*.yml`: 4 forzantes de CAMELS-COL
  más 15 de ERA5-Land y 47 atributos, `data/processed_era5land`). Corridas `espacial_era5l_p{k}_s{semilla}`.
- `transformer`: las mismas entradas, datos, particiones, épocas, pérdida y lote que `era5l`; solo cambia la
  arquitectura (`TRANSFORMER`). Es la comparación B3 del plan, en igualdad de condiciones con el LSTM.

Diseño (fijado antes de ver resultados; notebook de modelado, secciones 3, 4 y 7):
- Partición k: entrena con las cuencas fuera de la partición (1981–2010) y simula las retenidas en todo su
  registro (1981–2022). Ninguna cuenca retenida comparte sistema de ríos con una de entrenamiento (sección 4).
- Mismo protocolo que la confirmación C1 (sección 6.6): configuración de 6.1, 3 épocas fijas (la mejor época
  de 6.1 y de todas las corridas temporales), sin parada temprana, semillas 42, 43 y 44.
- La referencia de cuenca vista es la corrida C1 de la misma semilla (`configs/corridas/confirmacion_base_s*.yml`),
  entrenada con las 248 cuencas; si no existe en `runs/`, se entrena aquí.
- Cada corrida se entrena en un proceso aparte, que libera la RAM y la GPU al terminar. Si se corta, al volver a
  ejecutar se saltan las corridas que ya tienen su evaluación.
"""
import argparse
from collections.abc import Iterator
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DIR_CORRIDAS = ROOT / "runs"
DIR_CONFIGS = ROOT / "configs" / "corridas"
DIR_CUENCAS = ROOT / "configs" / "basins"
DIR_REGISTRO = DIR_CORRIDAS / "flujo_espacial"
N_PARTICIONES = 5
EPOCAS = 3
REGISTRO_COMPLETO = ("01/01/1981", "31/12/2022")
VARIANTES = ("base", "era5l", "transformer")
# Transformer de NeuralHydrology, fijado antes de ver resultados. Las entradas pasan por una capa de embedding
# (64 dinámicas + 64 estáticas = 128, divisible por 4 cabezas); 2 capas de codificador, como en los ejemplos de
# NeuralHydrology y Liu et al. (2025). Tasa de aprendizaje 1e-4: sin calentamiento, 5e-4 suele desestabilizar
# la atención. La capa de salida conserva el dropout de 0,4 del LSTM.
TRANSFORMER = {"model": "transformer", "transformer_nlayers": 2, "transformer_nheads": 4,
               "transformer_dim_feedforward": 256, "transformer_dropout": 0.1,
               "transformer_positional_encoding_type": "sum", "transformer_positional_dropout": 0.0,
               "dynamics_embedding": {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": 0.0},
               "statics_embedding": {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": 0.0},
               "learning_rate": 0.0001}


def config_base(variante: str = "base", semilla: int = 42) -> dict:
    """Configuración de partida: `configs/base.yml` (base), la C2 de la etapa 1 de esa semilla (era5l) o la C2 con
    el Transformer en lugar del LSTM (transformer)."""
    if variante == "base":
        return yaml.safe_load((ROOT / "configs" / "base.yml").read_text(encoding="utf-8"))
    c = yaml.safe_load((DIR_CONFIGS / f"confirmacion_era5l_s{semilla}.yml").read_text(encoding="utf-8"))
    if variante == "transformer":
        for k in ("hidden_size", "initial_forget_bias"):
            c.pop(k, None)
        c.update(TRANSFORMER)
    return c


def prefijo(variante: str) -> str:
    """Prefijo de los nombres de corrida de la variante."""
    return "espacial" if variante == "base" else f"espacial_{variante}"


def protocolo_c1(c: dict, semilla: int) -> dict:
    """El protocolo de la confirmación C1 (sección 6.6): 3 épocas fijas, sin parada temprana."""
    c = {**c, "seed": semilla, "epochs": EPOCAS, "early_stopping": False, "save_validation_results": False,
         "verbose": 0}
    for k in ("patience_early_stopping", "minimum_epochs_before_early_stopping",
              "patience_dynamic_learning_rate", "factor_dynamic_learning_rate"):
        c.pop(k, None)
    return c


def lista(nombre: str) -> list[str]:
    """Códigos de cuenca de un archivo de `configs/basins/`."""
    return [l for l in (DIR_CUENCAS / nombre).read_text().split() if l]


def config_particion(k: int, semilla: int, variante: str = "base") -> dict:
    """Configuración de la partición k: entrena sin sus cuencas y las simula en todo su registro."""
    c = protocolo_c1(config_base(variante, semilla), semilla)
    c.update(experiment_name=f"{prefijo(variante)}_p{k}_s{semilla}",
             train_basin_file=f"configs/basins/particion_{k}_entrenamiento.txt",
             validation_basin_file=f"configs/basins/particion_{k}_validacion.txt",
             test_basin_file=f"configs/basins/particion_{k}_prueba.txt",
             test_start_date=REGISTRO_COMPLETO[0], test_end_date=REGISTRO_COMPLETO[1],
             validate_n_random_basins=len(lista(f"particion_{k}_validacion.txt")))
    return c


def config_temporal(semilla: int, variante: str = "base") -> dict:
    """C1 (o C2, si la variante es era5l) de la confirmación (6.6), con la prueba 2016–2022 de las 183 cuencas."""
    ruta = DIR_CONFIGS / f"confirmacion_{variante}_s{semilla}.yml"
    if ruta.exists():
        return yaml.safe_load(ruta.read_text(encoding="utf-8"))
    c = protocolo_c1(config_base(), semilla)
    c.update(experiment_name=f"confirmacion_base_s{semilla}", save_validation_results=True)
    return c


def corrida(nombre: str) -> Path | None:
    """Carpeta más reciente de la corrida `nombre` que ya tiene los pesos de la última época, o None."""
    hechas = sorted(p for p in DIR_CORRIDAS.glob(f"{nombre}_*")
                    if p.name[len(nombre) + 1:].replace("_", "").isdigit() and (p / f"model_epoch{EPOCAS:03d}.pt").exists())
    return hechas[-1] if hechas else None


def evaluada(run_dir: Path | None) -> bool:
    """True si la corrida ya tiene su evaluación de prueba de la última época."""
    return run_dir is not None and (run_dir / "test" / f"model_epoch{EPOCAS:03d}" / "test_results.p").exists()


def ejecutar(args: list[str], log: Path) -> None:
    """Corre `nh_run` con esos argumentos en un proceso aparte y agrega su salida al log."""
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(args)}\n")
        f.flush()
        subprocess.run([sys.executable, "-m", "neuralhydrology.nh_run", *args],
                       stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=True)


def entrenar_y_evaluar(config: dict) -> Path:
    """Entrena (si hace falta) y evalúa una corrida; salta las ya evaluadas y anota el tiempo en progreso.log."""
    nombre = config["experiment_name"]
    run_dir = corrida(nombre)
    if evaluada(run_dir):
        print(f"{nombre}: ya está ({run_dir.name})", flush=True)
        return run_dir
    DIR_REGISTRO.mkdir(parents=True, exist_ok=True)
    log = DIR_REGISTRO / f"{nombre}.log"
    t0 = time.time()
    if run_dir is None:
        ruta = DIR_CONFIGS / f"{nombre}.yml"
        if not ruta.exists():   # no reescribir configuraciones existentes (p. ej., las de la etapa 1)
            encabezado = "# Generado por src/flujo_espacial.py (etapa 2, protocolo C1: 3 épocas fijas).\n"
            ruta.write_text(encabezado + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        ejecutar(["train", "--config-file", str(ruta)], log)
        run_dir = corrida(nombre)
        assert run_dir is not None, f"{nombre}: el entrenamiento no dejó los pesos de la época {EPOCAS}"
    t1 = time.time()
    ejecutar(["evaluate", "--run-dir", str(run_dir), "--period", "test", "--epoch", str(EPOCAS)], log)
    (run_dir / "terminado.txt").write_text(f"Entrenado y evaluado el {datetime.now():%Y-%m-%d %H:%M}\n")
    linea = (f"{datetime.now():%Y-%m-%d %H:%M} {nombre}: entrenamiento {(t1 - t0) / 60:.0f} min, "
             f"evaluación {(time.time() - t1) / 60:.0f} min ({run_dir.name})")
    with open(DIR_REGISTRO / "progreso.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")
    print(linea, flush=True)
    return run_dir


def plan(semillas: list[int], con_temporal: bool = True, variante: str = "base") -> Iterator[dict]:
    """Orden: por semilla, las 5 particiones y luego la referencia temporal, para que si la noche se corta queden
    semillas completas."""
    for s in semillas:
        for k in range(1, N_PARTICIONES + 1):
            yield config_particion(k, s, variante)
        if con_temporal:
            yield config_temporal(s, variante)


def main() -> None:
    """Línea de comandos: `entrenar` o `estado`, por variante y semillas."""
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["entrenar", "estado"])
    p.add_argument("--semillas", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--sin-temporal", action="store_true")
    p.add_argument("--variante", choices=VARIANTES, default="base")
    a = p.parse_args()
    for config in plan(a.semillas, not a.sin_temporal, a.variante):
        if a.accion == "estado":
            r = corrida(config["experiment_name"])
            print(f"{config['experiment_name']}: {'evaluada' if evaluada(r) else 'entrenada' if r else 'pendiente'}")
        else:
            entrenar_y_evaluar(config)


if __name__ == "__main__":
    main()
