"""Etapa 2: métricas de las cuencas no vistas (5 particiones) frente a las referencias y al modelo que vio el caudal.

Uso, desde la raíz del repositorio:  python src/resultados_espacial.py [--variante base|era5l]

Lee las corridas de `src/flujo_espacial.py` (época 3) y `runs/flujo_espacial/baselines.parquet`, y escribe en
`runs/flujo_espacial/`: `series_espacial.parquet` (observado y simulado por día), `metricas_cuencas.csv`,
`resumen.csv`, `comparaciones.csv` y `por_region.csv`. Copia las tablas chicas a `docs/flujo-espacial-*.csv`.
Con `--variante era5l` lee las corridas `espacial_era5l_*` y `confirmacion_era5l_*`, escribe en
`runs/flujo_espacial/era5l/` y `docs/flujo-espacial-era5l-*.csv`, y compara contra el LSTM base (debe existir
`runs/flujo_espacial/series_espacial.parquet` de la variante base).

Reglas de evaluación (las mismas de la sección 5 del notebook):
- días con caudal observado y simulado por el LSTM, con la simulación recortada en 0; todos los modelos se
  evalúan en esos mismos días;
- al menos 365 días válidos por cuenca;
- medianas entre cuencas.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from neuralhydrology.evaluation.metrics import calculate_metrics
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines_espacial import mercator_a_grados  # noqa: E402
from flujo_espacial import EPOCAS, N_PARTICIONES, VARIANTES, corrida, evaluada, prefijo  # noqa: E402

DIR = ROOT / "runs" / "flujo_espacial"
OBJETIVO = "streamflow"
METRICAS_NH = ["NSE", "KGE", "RMSE", "FHV", "FLV", "FMS"]
MIN_DIAS = 365
SEMILLAS = [42, 43, 44]
INICIO_TEST, FIN_TEST = "2016-01-01", "2022-12-31"
PRINCIPAL = "LSTM regional (ensamble)"   # nombre del modelo de la variante en las tablas
# Modelos de otras variantes que se agregan como referencia: nombre → (variante, columna de su series_espacial)
REFERENCIAS = {"era5l": {"LSTM regional base (forzantes CAMELS-COL)": ("base", "sim_lstm")},
               "transformer": {"LSTM ERA5-Land, semilla 42": ("era5l", "sim_lstm_s42"),
                               "LSTM ERA5-Land (ensamble)": ("era5l", "sim_lstm")}}


def leer_test(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Observado y simulado (recortado en 0) de cada cuenca en la evaluación de prueba de la última época."""
    with open(run_dir / "test" / f"model_epoch{EPOCAS:03d}" / "test_results.p", "rb") as f:
        res = pickle.load(f)
    salida = {}
    for g, r in res.items():
        ds = r["1D"]["xr"].isel(time_step=-1)
        salida[str(g)] = pd.DataFrame({"obs": ds[f"{OBJETIVO}_obs"].to_series(),
                                       "sim": ds[f"{OBJETIVO}_sim"].to_series().clip(lower=0)})
    return salida


def ensamble(lista_series: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Promedio de las simulaciones de varias semillas (mismos días por construcción)."""
    base = lista_series[0]
    return {g: base[g].assign(sim=np.mean([s[g]["sim"].to_numpy() for s in lista_series], axis=0)) for g in base}


def metricas_serie(obs: np.ndarray, sim: np.ndarray) -> dict | None:
    """Métricas de una cuenca en los días con los dos valores; None si son menos de MIN_DIAS."""
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < MIN_DIAS:
        return None
    o, s = obs[m], sim[m]
    idx = pd.date_range("2000-01-01", periods=m.sum())   # las métricas de NH solo usan el orden de los días
    fila = calculate_metrics(xr.DataArray(o, coords={"date": idx}, dims="date"),
                             xr.DataArray(s, coords={"date": idx}, dims="date"), metrics=METRICAS_NH, resolution="1D")
    fila.update(MAE=float(np.abs(o - s).mean()), r=float(np.corrcoef(o, s)[0, 1]) if s.std() > 0 else np.nan,
                alpha=float(s.std() / o.std()), beta=float(s.mean() / o.mean()), dias=int(m.sum()))
    return fila


def tabla_metricas(series: dict[str, pd.DataFrame], columna: str, desde: str | None = None,
                   hasta: str | None = None) -> pd.DataFrame:
    """Métricas por cuenca de un modelo, en los mismos días que el LSTM."""
    filas = {}
    for g, df in series.items():
        df = df.loc[desde:hasta]
        mascara = df["sim_lstm"].notna() & df["obs"].notna()      # mismos días que el LSTM
        f = metricas_serie(df["obs"].where(mascara).to_numpy(float), df[columna].where(mascara).to_numpy(float))
        if f is not None:
            filas[g] = f
    return pd.DataFrame.from_dict(filas, orient="index").rename_axis("gauge_id")


def resumir(t: pd.DataFrame) -> pd.Series:
    """Medianas entre cuencas y conteos de cuencas por rango de NSE."""
    return pd.Series({"cuencas": len(t), **t[["NSE", "KGE", "RMSE", "MAE", "FHV", "FLV", "r", "alpha", "beta"]].median(),
                      "NSE < 0": int((t["NSE"] < 0).sum()), "NSE ≥ 0,5": int((t["NSE"] >= 0.5).sum()),
                      "NSE ≥ 0,7": int((t["NSE"] >= 0.7).sum())})


def comparar(nombre: str, a: pd.DataFrame, b: pd.DataFrame, metrica: str = "NSE") -> dict:
    """b − a por cuenca (a es la referencia)."""
    d = (b[metrica] - a[metrica]).dropna()
    return {"comparación": nombre, "métrica": metrica, "cuencas": len(d), "referencia": a.loc[d.index, metrica].median(),
            "comparado": b.loc[d.index, metrica].median(), "dif. mediana": d.median(),
            "cuencas que mejoran": int((d > 0).sum()), "p (Wilcoxon)": wilcoxon(d).pvalue if (d != 0).any() else np.nan}


def main(variante: str = "base") -> None:
    """Calcula y guarda las tablas de la variante e imprime el resumen."""
    global PRINCIPAL
    PRINCIPAL = "Transformer regional" if variante == "transformer" else "LSTM regional (ensamble)"
    pref = prefijo(variante)
    salida = DIR if variante == "base" else DIR / variante
    docs = "flujo-espacial" if variante == "base" else f"flujo-espacial-{variante}"
    salida.mkdir(parents=True, exist_ok=True)
    # LSTM espacial: cada cuenca retenida la simula el modelo de su partición
    semillas = [s for s in SEMILLAS if all(evaluada(corrida(f"{pref}_p{k}_s{s}")) for k in range(1, N_PARTICIONES + 1))]
    assert semillas, "Ninguna semilla tiene las 5 particiones evaluadas"
    por_semilla = {}
    for s in semillas:
        por_semilla[s] = {}
        for k in range(1, N_PARTICIONES + 1):
            por_semilla[s].update(leer_test(corrida(f"{pref}_p{k}_s{s}")))
    lstm = ensamble([por_semilla[s] for s in semillas])

    # Referencia de cuenca vista: C1 (248 cuencas) de las mismas semillas, prueba 2016–2022
    sem_t = [s for s in semillas if evaluada(corrida(f"confirmacion_{variante}_s{s}"))]
    vista = ensamble([leer_test(corrida(f"confirmacion_{variante}_s{s}")) for s in sem_t]) if sem_t else {}
    # Variante distinta de base: el LSTM base (ensamble) como referencia directa
    # Variante distinta de base: modelos de otras variantes como referencia directa (REFERENCIAS)
    otras = {}
    for nombre, (var, col) in REFERENCIAS.get(variante, {}).items():
        d = pd.read_parquet((DIR if var == "base" else DIR / var) / "series_espacial.parquet",
                            columns=["gauge_id", "date", col])
        otras[nombre] = {str(g): x.set_index("date")[col] for g, x in d.groupby("gauge_id")}

    base = pd.read_parquet(DIR / "baselines.parquet")
    base["gauge_id"] = base["gauge_id"].astype(str)
    aux = pd.read_csv(ROOT / "data" / "processed" / "atributos.csv", dtype={"gauge_id": str}, index_col="gauge_id")
    aux["gauge_lat"], aux["gauge_lon"] = mercator_a_grados(aux["gauge_lat"], aux["gauge_lon"])

    series = {}
    for g, b in base.groupby("gauge_id"):
        b = b.set_index("date").sort_index()
        df = b[["particion", "obs", "clim_donantes", "vecino", "lgbm"] + (["idw"] if "idw" in b else [])].copy()
        df["sim_lstm"] = lstm[g]["sim"].reindex(df.index)
        for s in semillas:
            df[f"sim_lstm_s{s}"] = por_semilla[s][g]["sim"].reindex(df.index)
        df["promedio_lstm_lgbm"] = (df["sim_lstm"] + df["lgbm"]) / 2
        df["sim_vista"] = vista[g]["sim"].reindex(df.index) if g in vista else np.nan
        for i, nombre in enumerate(otras):
            df[f"sim_ref{i}"] = otras[nombre][g].reindex(df.index)
        # mismo caudal observado en ambas fuentes; rtol 1e-4 por el redondeo float32 de NeuralHydrology (~1e-5)
        assert np.allclose(df["obs"].dropna(), lstm[g]["obs"].reindex(df.index)[df["obs"].notna()],
                           rtol=1e-4, equal_nan=True)
        series[g] = df
    pd.concat(series, names=["gauge_id"]).reset_index().to_parquet(salida / "series_espacial.parquet", index=False)

    modelos = {PRINCIPAL: "sim_lstm", **{f"LSTM, semilla {s}": f"sim_lstm_s{s}" for s in semillas},
               "LightGBM regional": "lgbm", "Promedio LSTM + LightGBM": "promedio_lstm_lgbm",
               "Vecino más cercano (caudal observado)": "vecino", "Climatología de 5 donantes": "clim_donantes"}
    if "idw" in base:
        modelos["IDW de 5 estaciones (caudal observado)"] = "idw"
    for i, nombre in enumerate(otras):
        modelos[nombre] = f"sim_ref{i}"
    tablas = {(m, "1981–2022"): tabla_metricas(series, c) for m, c in modelos.items()}
    tablas.update({(m, "2016–2022"): tabla_metricas(series, c, INICIO_TEST, FIN_TEST)
                   for m, c in modelos.items() if "semilla" not in m})

    # Mismas cuencas y días de 2016–2022 con el modelo que vio el caudal: también deben tener sim_vista
    con_vista = {g: df.assign(sim_lstm=df["sim_lstm"].where(df["sim_vista"].notna())) for g, df in series.items()}
    tablas[(f"LSTM que vio el caudal (C1, {variante})", "2016–2022, cuencas con ambos")] = tabla_metricas(con_vista, "sim_vista", INICIO_TEST, FIN_TEST)
    tablas[(PRINCIPAL, "2016–2022, cuencas con ambos")] = tabla_metricas(con_vista, "sim_lstm", INICIO_TEST, FIN_TEST)

    resumen = pd.DataFrame({k: resumir(t) for k, t in tablas.items() if len(t)}).T.rename_axis(["modelo", "período"])
    for c in ["cuencas", "NSE < 0", "NSE ≥ 0,5", "NSE ≥ 0,7"]:
        resumen[c] = resumen[c].astype(int)

    L = tablas[(PRINCIPAL, "1981–2022")]
    comparaciones = [comparar("LSTM − climatología de donantes", tablas[("Climatología de 5 donantes", "1981–2022")], L),
                     comparar("LSTM − vecino más cercano", tablas[("Vecino más cercano (caudal observado)", "1981–2022")], L),
                     comparar("LSTM − LightGBM", tablas[("LightGBM regional", "1981–2022")], L),
                     comparar("Promedio − LSTM", L, tablas[("Promedio LSTM + LightGBM", "1981–2022")]),
                     comparar("LSTM − vecino más cercano (KGE)", tablas[("Vecino más cercano (caudal observado)", "1981–2022")], L, "KGE"),
                     comparar("LSTM − LightGBM (KGE)", tablas[("LightGBM regional", "1981–2022")], L, "KGE")]
    if "idw" in base:
        comparaciones.append(comparar("LSTM − IDW de 5 estaciones", tablas[("IDW de 5 estaciones (caudal observado)", "1981–2022")], L))
    for nombre in otras:
        B = tablas[(nombre, "1981–2022")]
        comparaciones += [comparar(f"{variante} − {nombre}", B, L), comparar(f"{variante} − {nombre} (KGE)", B, L, "KGE")]
    if len(tablas[(f"LSTM que vio el caudal (C1, {variante})", "2016–2022, cuencas con ambos")]):
        comparaciones.append(comparar("No vista − vista (2016–2022)",
                                      tablas[(f"LSTM que vio el caudal (C1, {variante})", "2016–2022, cuencas con ambos")],
                                      tablas[(PRINCIPAL, "2016–2022, cuencas con ambos")]))
    comparaciones = pd.DataFrame(comparaciones)

    region = aux["region"]
    por_region = pd.DataFrame({m: tablas[(m, "1981–2022")]["NSE"].groupby(region).median()
                               for m in [PRINCIPAL, "LightGBM regional",
                                         "Vecino más cercano (caudal observado)", "Climatología de 5 donantes"]})
    por_region.insert(0, "cuencas", L.groupby(region).size())
    particion = base.drop_duplicates("gauge_id").set_index("gauge_id")["particion"]
    por_particion = pd.DataFrame({"cuencas": L.groupby(particion).size(), "NSE LSTM": L["NSE"].groupby(particion).median(),
                                  "KGE LSTM": L["KGE"].groupby(particion).median()})

    largo = pd.concat({f"{m} | {p}": t for (m, p), t in tablas.items()}, names=["modelo"]).reset_index()
    largo.to_csv(salida / "metricas_cuencas.csv", index=False)
    resumen.to_csv(salida / "resumen.csv")
    comparaciones.to_csv(salida / "comparaciones.csv", index=False)
    por_region.to_csv(salida / "por_region.csv")

    # Copias chicas, versionables
    porcuenca = pd.DataFrame({"region": region.reindex(L.index), "particion": particion.reindex(L.index),
                              "area_km2": aux["area"].reindex(L.index), "lat": aux["gauge_lat"].reindex(L.index),
                              "lon": aux["gauge_lon"].reindex(L.index),
                              **{f"NSE_{c}": tablas[(m, "1981–2022")]["NSE"] for m, c in modelos.items() if "semilla" not in m},
                              "KGE_sim_lstm": L["KGE"], "FHV_sim_lstm": L["FHV"], "beta_sim_lstm": L["beta"],
                              "dias": L["dias"]}).round(4)
    porcuenca.to_csv(ROOT / "docs" / f"{docs}-cuencas.csv")
    resumen.round(3).to_csv(ROOT / "docs" / f"{docs}-resumen.csv")
    comparaciones.round(4).to_csv(ROOT / "docs" / f"{docs}-comparaciones.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Variante: {variante} | semillas: {semillas} | referencia vista: {sem_t}\n")
    print(resumen.round(3).to_string(), "\n")
    print(comparaciones.round(4).to_string(), "\n")
    print(por_region.round(3).to_string(), "\n")
    print(por_particion.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variante", choices=VARIANTES, default="base")
    main(ap.parse_args().variante)
