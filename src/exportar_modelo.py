"""Despliegue (fase C del plan): el LSTM de NeuralHydrology como un modelo autónomo que recibe entradas crudas.

Uso, desde la raíz del repositorio:
    python src/exportar_modelo.py verificar --corrida espacial_era5l_p1_s42
    python src/exportar_modelo.py exportar --variante era5l --semillas 42 43 44

`Envoltura` recibe los forzantes de 365 días y los atributos de la cuenca **sin normalizar**, los normaliza con el
escalador guardado de la corrida (`train_data/train_data_scaler.yml`: media y desviación de las cuencas de
entrenamiento), corre el LSTM y devuelve el caudal del último día en mm/día, recortado en 0. Es lo mismo que hace
NeuralHydrology al evaluar, así que la aplicación no necesita NeuralHydrology ni el escalador aparte.

- `verificar` simula con la envoltura las cuencas de prueba de una corrida y compara con lo que guardó
  NeuralHydrology en su evaluación (`test_results.p`).
- `exportar` guarda cada modelo en ONNX (`despliegue/modelos/`) con un JSON de metadatos (entradas en orden, cuencas
  retenidas de la partición) y comprueba con onnxruntime que da lo mismo que PyTorch.

El caudal observado no entra en ningún lado: `streamflow` está en los archivos de datos, pero no en las entradas.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flujo_espacial import EPOCAS, N_PARTICIONES, corrida, lista  # noqa: E402

SALIDA = ROOT / "despliegue" / "modelos"


class Envoltura(torch.nn.Module):
    """LSTM de NeuralHydrology con la normalización de entradas y salida incluida."""

    def __init__(self, run_dir: Path):
        super().__init__()
        cfg = Config(run_dir / "config.yml")
        self.dinamicas: list[str] = list(cfg.dynamic_inputs)
        # NeuralHydrology ordena los atributos alfabéticamente antes de armar x_s (basedataset.py, sort_index)
        self.estaticas: list[str] = sorted(cfg.static_attributes)
        self.seq_length: int = cfg.seq_length
        self.modelo = get_model(cfg).eval()
        pesos = torch.load(run_dir / f"model_epoch{EPOCAS:03d}.pt", map_location="cpu", weights_only=True)
        self.modelo.load_state_dict(pesos)
        esc = load_scaler(run_dir)
        centro, escala = esc["xarray_feature_center"], esc["xarray_feature_scale"]
        f32 = lambda v: torch.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731
        self.register_buffer("d_centro", f32([centro[v].item() for v in self.dinamicas]))
        self.register_buffer("d_escala", f32([escala[v].item() for v in self.dinamicas]))
        self.register_buffer("s_media", f32(esc["attribute_means"][self.estaticas].to_numpy()))
        self.register_buffer("s_desv", f32(esc["attribute_stds"][self.estaticas].to_numpy()))
        objetivo = cfg.target_variables[0]
        self.register_buffer("y_centro", f32(centro[objetivo].item()))
        self.register_buffer("y_escala", f32(escala[objetivo].item()))
        self.eval()   # siempre en modo evaluación (sin dropout); torch.onnx.export restaura el modo al terminar

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        """x_d: [lote, 365, n dinámicas] crudas; x_s: [lote, n estáticas] crudas → caudal [lote] en mm/día."""
        x_d = (x_d - self.d_centro) / self.d_escala
        x_s = (x_s - self.s_media) / self.s_desv
        datos = {"x_d": {v: x_d[:, :, i:i + 1] for i, v in enumerate(self.dinamicas)}, "x_s": x_s}
        y = self.modelo(datos)["y_hat"][:, -1, 0]
        return torch.clamp(y * self.y_escala + self.y_centro, min=0.0)


def entradas_cuenca(run_dir: Path, gauge_id: str, env: Envoltura) -> tuple[pd.DataFrame, np.ndarray]:
    """Forzantes diarios (sin el caudal) y atributos crudos de una cuenca, de la base de datos de la corrida."""
    cfg = Config(run_dir / "config.yml")
    ds = xr.open_dataset(Path(cfg.data_dir) / "time_series" / f"{gauge_id}.nc")
    forzantes = ds[env.dinamicas].to_dataframe()
    atributos = pd.read_csv(Path(cfg.data_dir) / "attributes" / "atributos_modelo.csv", dtype={"gauge_id": str},
                            index_col="gauge_id").loc[gauge_id, env.estaticas].to_numpy(np.float32)
    return forzantes, atributos


def simular(env: Envoltura, forzantes: pd.DataFrame, atributos: np.ndarray, desde: str, hasta: str,
            lote: int = 512) -> pd.Series:
    """Caudal simulado de cada día de [desde, hasta] con ventanas de 365 días; NaN si la ventana tiene huecos."""
    x = forzantes.to_numpy(np.float32)
    fechas = forzantes.index
    objetivo = np.where((fechas >= desde) & (fechas <= hasta))[0]
    objetivo = objetivo[objetivo >= env.seq_length - 1]
    ventanas = np.lib.stride_tricks.sliding_window_view(x, env.seq_length, axis=0).transpose(0, 2, 1)
    ventanas = ventanas[objetivo - env.seq_length + 1]
    completas = np.isfinite(ventanas).all(axis=(1, 2))
    sim = np.full(len(objetivo), np.nan, dtype=np.float32)
    idx = np.where(completas)[0]
    with torch.no_grad():
        for i in range(0, len(idx), lote):
            j = idx[i:i + lote]
            xs = torch.from_numpy(np.repeat(atributos[None], len(j), axis=0))
            sim[j] = env(torch.from_numpy(np.ascontiguousarray(ventanas[j])), xs).numpy()
    return pd.Series(sim, index=fechas[objetivo], name="sim")


def verificar(nombre: str) -> pd.DataFrame:
    """Compara la envoltura con la evaluación de NeuralHydrology en las cuencas de prueba de la corrida."""
    run_dir = corrida(nombre)
    env = Envoltura(run_dir)
    cfg = Config(run_dir / "config.yml")
    with open(run_dir / "test" / f"model_epoch{EPOCAS:03d}" / "test_results.p", "rb") as f:
        res = pickle.load(f)
    filas = []
    for g, r in res.items():
        nh = r["1D"]["xr"].isel(time_step=-1)[f"{cfg.target_variables[0]}_sim"].to_series().clip(lower=0)
        forz, attr = entradas_cuenca(run_dir, str(g), env)
        mia = simular(env, forz, attr, str(nh.index.min().date()), str(nh.index.max().date()))
        comun = nh.dropna().index.intersection(mia.dropna().index)
        dif = np.abs(nh.loc[comun].to_numpy() - mia.loc[comun].to_numpy())
        filas.append({"gauge_id": str(g), "días": len(comun), "días solo en NH": len(nh.dropna().index.difference(comun)),
                      "dif. máx. (mm/día)": float(dif.max()), "dif. media (mm/día)": float(dif.mean())})
    return pd.DataFrame(filas).set_index("gauge_id")


def exportar(variante: str, semillas: list[int]) -> None:
    """Guarda en ONNX el modelo de cada partición y semilla, con sus metadatos, y lo comprueba con onnxruntime."""
    import onnxruntime as ort
    SALIDA.mkdir(parents=True, exist_ok=True)
    pref = "espacial" if variante == "base" else f"espacial_{variante}"
    for s in semillas:
        for k in range(1, N_PARTICIONES + 1):
            nombre = f"{pref}_p{k}_s{s}"
            env = Envoltura(corrida(nombre))
            x_d = torch.rand(4, env.seq_length, len(env.dinamicas)) * 5
            x_s = env.s_media.repeat(4, 1)
            ruta = SALIDA / f"{nombre}.onnx"
            torch.onnx.export(env, (x_d, x_s), ruta, input_names=["forzantes", "atributos"], output_names=["caudal"],
                              dynamic_axes={"forzantes": {0: "lote"}, "atributos": {0: "lote"}, "caudal": {0: "lote"}},
                              opset_version=17, dynamo=False)
            env.eval()
            sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
            y_onnx = sesion.run(None, {"forzantes": x_d.numpy(), "atributos": x_s.numpy()})[0]
            with torch.no_grad():
                y_torch = env(x_d, x_s).numpy()
            assert np.allclose(y_onnx, y_torch, rtol=1e-4, atol=1e-4), (nombre, y_onnx, y_torch)
            meta = {"corrida": nombre, "particion": k, "semilla": s, "dias_de_entrada": env.seq_length,
                    "forzantes": env.dinamicas, "atributos": env.estaticas, "unidad_salida": "mm/día",
                    "cuencas_no_vistas": lista(f"particion_{k}_prueba.txt")}
            (SALIDA / f"{nombre}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{nombre}: {ruta.stat().st_size / 1e6:.1f} MB, ONNX = PyTorch (dif. máx. "
                  f"{np.abs(y_onnx - y_torch).max():.1e})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("accion", choices=["verificar", "exportar"])
    ap.add_argument("--corrida", default="espacial_era5l_p1_s42")
    ap.add_argument("--variante", default="era5l")
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    if a.accion == "verificar":
        t = verificar(a.corrida)
        print(t.round(5).to_string())
        print(f"\nDiferencia máxima en todas las cuencas: {t['dif. máx. (mm/día)'].max():.2e} mm/día")
    else:
        exportar(a.variante, a.semillas)
