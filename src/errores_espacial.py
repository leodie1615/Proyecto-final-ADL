"""Etapa 2: análisis de errores en las cuencas no vistas (mejoras B7 y B8 del plan). Uso:
    python src/errores_espacial.py [--variante era5l]

B7, crecientes por período de retorno. Para cada cuenca retenida se ajusta una distribución de Gumbel (por momentos,
estable aun con máximos casi iguales, como los de la climatología) a los máximos anuales del caudal observado (años con al menos 300 días con dato) y se obtienen los umbrales de 2, 5 y 10 años. Un
evento observado es una racha de días sobre el umbral, separada de la siguiente por más de 7 días; se representa
por su día de máximo. Cada modelo se evalúa de dos formas:
- **umbral observado:** el modelo debe superar el mismo umbral, en mm/día, que el observado;
- **umbral propio:** el umbral sale de los máximos anuales del propio modelo. Es lo que se podría hacer en una cuenca
  sin estación (Nearing et al., 2024: `calculate_return_period_metrics`), y descuenta el sesgo de amplitud.
Un evento observado es un acierto si el modelo tiene un evento con máximo a ±2 días. Se reportan el recall (eventos
detectados / observados), la precisión (eventos del modelo que coinciden / eventos del modelo) y F1, sumando los
eventos de todas las cuencas. El caudal observado solo se usa para evaluar.

B8, correlación de errores entre cuencas. Con los residuos diarios (simulado − observado) del modelo, se calcula la
correlación de Pearson entre cada par de cuencas retenidas con al menos 365 días en común, y se resume por distancia
entre estaciones y según estén o no en el mismo sistema de ríos. Si los errores de cuencas cercanas se parecen, el
modelo falla por algo regional (la lluvia, por ejemplo) y no por la cuenca.

Escribe `docs/flujo-espacial[-variante]-crecientes.csv` y `docs/flujo-espacial[-variante]-residuos.csv`.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "runs" / "flujo_espacial"
PERIODOS_RETORNO = (2, 5, 10)
MIN_DIAS_ANIO = 300
SEPARACION_EVENTOS = 7      # días entre rachas para que cuenten como eventos distintos
VENTANA = 2                 # ± días para emparejar un evento observado con uno simulado
MIN_DIAS_COMUNES = 365
BORDES_KM = [0, 50, 100, 200, 400, np.inf]
MODELOS = {"sim_lstm": "LSTM (ensamble)", "sim_ref0": "Referencia de otra variante", "lgbm": "LightGBM regional",
           "vecino": "Vecino más cercano", "clim_donantes": "Climatología de donantes"}


def umbrales(q: pd.Series) -> dict[int, float] | None:
    """Caudal de los períodos de retorno con una Gumbel ajustada a los máximos anuales; None si hay menos de 10 años."""
    anual = q.groupby(q.index.year)
    maximos = anual.max()[anual.count() >= MIN_DIAS_ANIO]
    if len(maximos) < 10:
        return None
    escala = float(maximos.std()) * np.sqrt(6) / np.pi          # Gumbel por momentos
    loc = float(maximos.mean()) - np.euler_gamma * escala
    return {t: float(gumbel_r.ppf(1 - 1 / t, loc, escala)) if escala > 0 else loc for t in PERIODOS_RETORNO}


def eventos(q: pd.Series, umbral: float) -> pd.DatetimeIndex:
    """Día de máximo de cada racha sobre el umbral (rachas a menos de SEPARACION_EVENTOS días se unen)."""
    sobre = q[q >= umbral]
    if sobre.empty:
        return pd.DatetimeIndex([])
    grupo = (sobre.index.to_series().diff().dt.days > SEPARACION_EVENTOS).cumsum()
    return pd.DatetimeIndex([g.idxmax() for _, g in sobre.groupby(grupo.to_numpy())])


def emparejar(observados: pd.DatetimeIndex, simulados: pd.DatetimeIndex) -> tuple[int, int]:
    """(eventos observados detectados, eventos simulados que coinciden con uno observado), con ±VENTANA días."""
    if len(observados) == 0 or len(simulados) == 0:
        return 0, 0
    d = np.abs((observados.values[:, None] - simulados.values[None, :]).astype("timedelta64[D]").astype(int))
    return int((d.min(axis=1) <= VENTANA).sum()), int((d.min(axis=0) <= VENTANA).sum())


def crecientes(series: pd.DataFrame, modelos: dict[str, str]) -> pd.DataFrame:
    """Aciertos, falsas alarmas, recall, precisión y F1 por modelo, período de retorno y tipo de umbral."""
    filas = []
    for g, df in series.groupby("gauge_id"):
        df = df.set_index("date")
        df = df[df["obs"].notna() & df["sim_lstm"].notna()]      # los mismos días para todos los modelos
        u_obs = umbrales(df["obs"])
        if u_obs is None:
            continue
        for col in modelos:
            u_mod = umbrales(df[col].dropna())
            for t in PERIODOS_RETORNO:
                ev_obs = eventos(df["obs"], u_obs[t])
                for tipo, u in (("umbral observado", u_obs[t]), ("umbral propio", u_mod[t] if u_mod else np.nan)):
                    ev_sim = eventos(df[col].dropna(), u) if np.isfinite(u) else pd.DatetimeIndex([])
                    aciertos, coinciden = emparejar(ev_obs, ev_sim)
                    filas.append({"gauge_id": g, "modelo": modelos[col], "T (años)": t, "umbral": tipo,
                                  "eventos observados": len(ev_obs), "eventos del modelo": len(ev_sim),
                                  "aciertos": aciertos, "coinciden": coinciden})
    f = pd.DataFrame(filas)
    total = f.groupby(["umbral", "T (años)", "modelo"], sort=False)[
        ["eventos observados", "eventos del modelo", "aciertos", "coinciden"]].sum()
    total["recall"] = total["aciertos"] / total["eventos observados"]
    total["precisión"] = total["coinciden"] / total["eventos del modelo"].replace(0, np.nan)
    total["F1"] = 2 * total["recall"] * total["precisión"] / (total["recall"] + total["precisión"])
    total["cuencas"] = f.groupby(["umbral", "T (años)", "modelo"], sort=False)["gauge_id"].nunique()
    return total.round(3)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distancia en km sobre una esfera de 6371 km."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def residuos(series: pd.DataFrame, cuencas: pd.DataFrame, sistema: pd.Series, columnas: dict[str, str]) -> pd.DataFrame:
    """Correlación mediana de los residuos entre pares de cuencas, por rango de distancia y por sistema de ríos."""
    filas = []
    for col, nombre in columnas.items():
        r = series.assign(res=series[col] - series["obs"]).pivot(index="date", columns="gauge_id", values="res")
        validos = r.notna().astype(float)
        comunes = validos.T @ validos                                   # días en común por par
        corr = r.corr(min_periods=MIN_DIAS_COMUNES)
        for a, b in combinations(r.columns, 2):
            if comunes.loc[a, b] < MIN_DIAS_COMUNES or not np.isfinite(corr.loc[a, b]):
                continue
            d = haversine_km(cuencas.loc[a, "lat"], cuencas.loc[a, "lon"], cuencas.loc[b, "lat"], cuencas.loc[b, "lon"])
            filas.append({"modelo": nombre, "a": a, "b": b, "km": float(d), "r": float(corr.loc[a, b]),
                          "mismo sistema": bool(sistema[a] == sistema[b])})
    pares = pd.DataFrame(filas)
    pares["distancia"] = pd.cut(pares["km"], BORDES_KM, right=False,
                                labels=["< 50 km", "50–100 km", "100–200 km", "200–400 km", "≥ 400 km"])
    resumen = pares.groupby(["modelo", "mismo sistema", "distancia"], observed=True)["r"].agg(
        pares="size", r_mediana="median", fraccion_r_mayor_0_3=lambda x: (x > 0.3).mean())
    return resumen.round(3)


def main(variante: str = "era5l") -> None:
    """Calcula y guarda las tablas de crecientes y de residuos de la variante."""
    sufijo = "" if variante == "base" else f"-{variante}"
    directorio = DIR if variante == "base" else DIR / variante
    series = pd.read_parquet(directorio / "series_espacial.parquet")
    series["gauge_id"] = series["gauge_id"].astype(str)
    modelos = {c: n for c, n in MODELOS.items() if c in series}
    if "sim_ref0" in modelos:
        modelos["sim_ref0"] = {"era5l": "LSTM base (ensamble)", "transformer": "LSTM ERA5-Land, semilla 42"}[variante]
    if variante == "transformer":
        modelos["sim_lstm"] = "Transformer"

    tabla = crecientes(series, modelos)
    tabla.to_csv(ROOT / "docs" / f"flujo-espacial{sufijo}-crecientes.csv")

    cuencas = pd.read_csv(ROOT / "docs" / f"flujo-espacial{sufijo}-cuencas.csv", dtype={"gauge_id": str},
                          index_col="gauge_id")
    sistema = pd.read_csv(ROOT / "configs" / "basins" / "particiones.csv", dtype={"gauge_id": str},
                          index_col="gauge_id")["sistema"]
    res = residuos(series, cuencas, sistema, {"sim_lstm": modelos["sim_lstm"], "lgbm": "LightGBM regional"})
    res.to_csv(ROOT / "docs" / f"flujo-espacial{sufijo}-residuos.csv")

    pd.set_option("display.width", 200)
    print(f"Variante: {variante}\n\nCrecientes (eventos sumados en todas las cuencas):\n{tabla.to_string()}\n")
    print(f"Correlación de los residuos entre pares de cuencas:\n{res.to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variante", choices=("base", "era5l", "transformer"), default="era5l")
    main(ap.parse_args().variante)
