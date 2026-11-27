"""Etapa 2: referencias para las cuencas retenidas de cada partición espacial (sin caudal de la cuenca retenida).

Uso, desde la raíz del repositorio:  python src/baselines_espacial.py

Escribe runs/flujo_espacial/baselines.parquet, con una fila por cuenca retenida y día (1981–2022):
- `clim_donantes`: climatología (1981–2010) promedio de las 5 cuencas de entrenamiento más parecidas en los 48
  atributos estandarizados (notebook de modelado, sección 8). Regionalización por similitud física.
- `vecino`: caudal diario observado (mm/día) de la estación de entrenamiento más cercana con dato ese día, entre las
  5 más cercanas por la distancia entre aforos. Regionalización por proximidad espacial (transferencia por área,
  porque el caudal ya está en mm/día). Usa caudal observado de otras cuencas en los mismos días: es la referencia
  clásica más exigente, y no cumple la regla del proyecto para el modelo, por eso solo es referencia.
- `lgbm`: LightGBM con las 83 variables de la sección 6.2 (forzantes, medias, índices antecedentes, día del año y
  atributos), entrenado con las cuencas de entrenamiento de la partición, como el LSTM.

Las cuencas de entrenamiento de cada partición nunca comparten sistema de ríos con las retenidas (sección 4).
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR_BASE = ROOT / "data" / "processed"
DIR_CUENCAS = ROOT / "configs" / "basins"
SALIDA = ROOT / "runs" / "flujo_espacial" / "baselines.parquet"
OBJETIVO = "streamflow"
FORZANTES = ["pr", "poten_evapo", "t_min", "t_max"]
N_PARTICIONES, N_DONANTES = 5, 5
INICIO_TRAIN, FIN_TRAIN = "1981-01-01", "2010-12-31"
DIAS_DEL_ANIO = pd.date_range("2000-01-01", "2000-12-31").strftime("%m-%d")

# Igual que el notebook de modelado, sección 6.2
PARAMETROS_LGBM = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                   "num_threads": 6, "seed": 42, "verbose": -1}
VENTANAS_LLUVIA = [2, 3, 5, 7, 15, 30, 60, 90, 180, 365]
VIDAS_MEDIAS = [5, 20, 60, 180]


def variables_lgbm(df):
    """Copia de `variables_lgbm` del notebook (6.2): solo forzantes del día y de los anteriores."""
    p, e, tn, tx = df["pr"], df["poten_evapo"], df["t_min"], df["t_max"]
    media = lambda s, w: s.rolling(w, min_periods=max(1, w // 2)).mean()
    x = {"pr": p, "pet": e, "t_min": tn, "t_max": tx}
    x.update({f"pr_dia_menos_{k}": p.shift(k) for k in [1, 2, 3]})
    x.update({f"pr_media_{w}": media(p, w) for w in VENTANAS_LLUVIA})
    x.update({f"pet_media_{w}": media(e, w) for w in [7, 30, 90, 365]})
    x.update({f"balance_{w}": media(p - e, w) for w in [30, 90, 180, 365]})
    x.update({f"tmax_media_{w}": media(tx, w) for w in [7, 30]})
    x.update({f"tmin_media_{w}": media(tn, w) for w in [7, 30]})
    x.update({f"pr_antecedente_{h}": p.ewm(halflife=h).mean() for h in VIDAS_MEDIAS})
    dia = df.index.dayofyear
    x["sin_dia"], x["cos_dia"] = np.sin(2 * np.pi * dia / 365.25), np.cos(2 * np.pi * dia / 365.25)
    return pd.DataFrame(x, index=df.index)


def climatologia(q):
    media = q.groupby(q.index.strftime("%m-%d")).mean().reindex(DIAS_DEL_ANIO)
    circular = pd.concat([media.iloc[-15:], media, media.iloc[:15]])
    return circular.rolling(31, center=True, min_periods=1).mean().iloc[15:-15]


def lista(nombre):
    return [l for l in (DIR_CUENCAS / nombre).read_text().split() if l]


def mercator_a_grados(y, x):
    """`gauge_lat` y `gauge_lon` de CAMELS-COL no son grados: son metros de World Mercator elipsoidal (EPSG:3395),
    el mismo sistema del shapefile (bitácora, sección 9). Inversa elipsoidal (WGS 84), por iteración."""
    a, e = 6378137.0, 0.0818191908426215
    lon = np.degrees(np.asarray(x, float) / a)
    t = np.exp(-np.asarray(y, float) / a)
    lat = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(10):
        lat = np.pi / 2 - 2 * np.arctan(t * ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2))
    return np.degrees(lat), lon


def distancia_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    atributos = pd.read_csv(DIR_BASE / "attributes" / "atributos_modelo.csv", dtype={"gauge_id": str},
                            index_col="gauge_id")
    aux = pd.read_csv(DIR_BASE / "atributos.csv", dtype={"gauge_id": str}, index_col="gauge_id")
    aux["gauge_lat"], aux["gauge_lon"] = mercator_a_grados(aux["gauge_lat"], aux["gauge_lon"])
    assert aux["gauge_lat"].between(-5, 13).all() and aux["gauge_lon"].between(-80, -66).all()
    tabla = pd.read_parquet(DIR_BASE / "camels_col_diario.parquet",
                            columns=["gauge_id", "date", "periodo", OBJETIVO] + FORZANTES)
    tabla["gauge_id"] = tabla["gauge_id"].astype(str)
    caudal = tabla.pivot(index="date", columns="gauge_id", values=OBJETIVO)
    clim = pd.DataFrame({g: climatologia(caudal.loc[INICIO_TRAIN:FIN_TRAIN, g]) for g in caudal.columns})
    claves = caudal.index.strftime("%m-%d")

    # Variables de LightGBM de todas las cuencas y días (se calculan una vez)
    X, y, periodo, sigma = {}, {}, {}, {}
    for g, df in tabla.groupby("gauge_id", sort=True):
        df = df.set_index("date").sort_index()
        X[g] = variables_lgbm(df).assign(**atributos.loc[g].to_dict()).astype("float32")
        y[g], periodo[g] = df[OBJETIVO], df["periodo"]
        sigma[g] = y[g][periodo[g] == "train"].std()
    nombres = list(next(iter(X.values())).columns)
    validan = set(lista("validacion.txt"))
    inicio = pd.Timestamp(INICIO_TRAIN)

    partes = []
    for k in range(1, N_PARTICIONES + 1):
        prueba, entrenamiento = lista(f"particion_{k}_prueba.txt"), lista(f"particion_{k}_entrenamiento.txt")
        assert not set(prueba) & set(entrenamiento)

        # LightGBM con las cuencas de entrenamiento (1981–2010, días alternos), parada con su validación 2011–2015
        def datos(conjunto, referencia=None):
            xs, ys, ws = [], [], []
            for g in entrenamiento:
                p, q = periodo[g], y[g]
                if conjunto == "train":
                    m = (p == "train") & q.notna() & ((q.index - inicio).days % 2 == 0)
                else:
                    m = (p == "validation") & q.notna() & (g in validan)
                if m.any():
                    xs.append(X[g][m].to_numpy()); ys.append(q[m].to_numpy())
                    ws.append(np.full(m.sum(), 1 / (sigma[g] + 0.1) ** 2))
            return lgb.Dataset(np.vstack(xs), np.concatenate(ys), weight=np.concatenate(ws), feature_name=nombres,
                               reference=referencia, free_raw_data=True)

        d_tr = datos("train")
        modelo = lgb.train(PARAMETROS_LGBM, d_tr, num_boost_round=3000, valid_sets=[datos("validation", d_tr)],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
        modelo.save_model(str(SALIDA.parent / f"lgbm_espacial_p{k}.txt"), num_iteration=modelo.best_iteration)
        print(f"partición {k}: LightGBM con {modelo.best_iteration} árboles", flush=True)

        # Climatología de donantes por similitud de atributos (sección 8 del notebook)
        ref = atributos.loc[entrenamiento]
        z = (atributos - ref.mean()) / ref.std()
        for g in prueba:
            d_attr = ((z.loc[entrenamiento] - z.loc[g]) ** 2).sum(axis=1)
            clim_g = clim[d_attr.nsmallest(N_DONANTES).index].mean(axis=1)

            # Vecino más cercano por distancia entre aforos, con dato ese día (hasta 5)
            d_geo = pd.Series(distancia_km(aux.at[g, "gauge_lat"], aux.at[g, "gauge_lon"],
                                           aux.loc[entrenamiento, "gauge_lat"].to_numpy(),
                                           aux.loc[entrenamiento, "gauge_lon"].to_numpy()), index=entrenamiento)
            vecinos = d_geo.nsmallest(N_DONANTES).index
            vecino = caudal[vecinos[0]].copy()
            for v in vecinos[1:]:
                vecino = vecino.fillna(caudal[v])

            pred = np.clip(modelo.predict(X[g].to_numpy(), num_iteration=modelo.best_iteration), 0, None)
            partes.append(pd.DataFrame({
                "gauge_id": g, "particion": k, "date": caudal.index, "obs": caudal[g].to_numpy(),
                "clim_donantes": clim_g.loc[claves].to_numpy(), "vecino": vecino.to_numpy(), "lgbm": pred,
                "dist_vecino_km": d_geo[vecinos[0]]}))
    salida = pd.concat(partes, ignore_index=True)
    salida.to_parquet(SALIDA, index=False)
    print(f"{SALIDA.relative_to(ROOT)}: {salida['gauge_id'].nunique()} cuencas, {len(salida):,} filas")


if __name__ == "__main__":
    main()
