"""Pipeline de preparación de CAMELS-COL (§4 del handoff).

Lee las series diarias y los atributos de cada cuenca y los consolida en una tabla.
La estandarización (z-score) NO se hace aquí: NeuralHydrology la calcula en cada
corrida usando solo las cuencas y el período de entrenamiento, lo que evita la fuga.

Salidas en data/processed/:
  camels_col_diario.parquet  Tabla única: una fila por cuenca y día (1981-2022) con
                             forzantes, caudal y los atributos estáticos del modelo.
  atributos.csv              Una fila por cuenca: atributos del modelo, auxiliares y firmas.
  resumen_cuencas.csv        Control de calidad por cuenca (vacíos, cobertura por período).
  resumen_interpolacion.csv  Días vacíos e interpolados por variable, sumando todas las cuencas.
  variables_finales.yml      Lista final de entradas, para configurar el modelo.

Uso:  python src/preprocess.py [--config configs/variables.yml]
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Archivo de atributos -> columnas a renombrar.
# El 08 (índices climáticos) no se lee: se recalculan con el período de entrenamiento.
ARCHIVOS_ATRIBUTOS = {
    "02_CAMELS_COL_Catchment_information.csv": {},
    "05_CAMELS_COL_Geologic_characteristics.csv": {},
    "06_CAMELS_COL_Land_cover_characteristics.csv": {"water_bodies_perc": "water_bodies_perc_cob"},
    "07_CAMELS_COL_Soil_characteristics.csv": {"water_bodies_perc": "water_bodies_perc_suelo"},
    "09_CAMELS_COL_Hydrological_signatures.csv": {},
    "10_CAMELS_COL_Physiograpic_characteristics.csv": {},
    "11_CAMELS_COL_Land_use_capability.csv": {},
}


# --------------------------------------------------------------------------- lectura

def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de atributos tolerando las inconsistencias del dataset:
    codificación UTF-8 o Latin-1, separador ';' o ',', espacios en los nombres de
    columna, filas vacías de relleno y gauge_id con decimales ("11017010.00")."""
    crudo = ruta.read_bytes()
    try:
        texto = crudo.decode("utf-8")
    except UnicodeDecodeError:
        texto = crudo.decode("latin-1")
    encabezado = texto.splitlines()[0]
    sep = ";" if encabezado.count(";") > encabezado.count(",") else ","

    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype={"gauge_id": str})
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["gauge_id"])
    df["gauge_id"] = df["gauge_id"].str.strip().str.replace(r"\.0+$", "", regex=True)
    if df["gauge_id"].duplicated().any():
        raise ValueError(f"gauge_id repetidos en {ruta.name}")
    return df.set_index("gauge_id")


def cargar_atributos(dir_atributos: Path, ruta_reporte: Path, ids: list[str]) -> pd.DataFrame:
    """Une todos los archivos de atributos en una tabla con una fila por cuenca."""
    partes = [leer_csv(dir_atributos / nombre).rename(columns=renombres)
              for nombre, renombres in ARCHIVOS_ATRIBUTOS.items()]
    attrs = pd.concat(partes, axis=1, join="outer")

    repetidas = attrs.columns[attrs.columns.duplicated()].tolist()
    if repetidas:
        raise ValueError(f"Columnas con el mismo nombre en varios archivos: {repetidas}")
    faltantes = sorted(set(ids) - set(attrs.index))
    if faltantes:
        raise ValueError(f"Cuencas sin atributos: {faltantes}")

    attrs = attrs.loc[ids].copy()
    reporte = pd.read_csv(ruta_reporte, dtype={"gauge_id": str}).set_index("gauge_id")
    attrs["region"] = reporte["region"].reindex(attrs.index)
    for col in ["gauge_star", "gauge_end"]:
        attrs[col] = pd.to_datetime(attrs[col], format="%d/%m/%Y")
    return attrs


def leer_serie(ruta: Path, cfg: dict, calendario: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Lee la serie de una cuenca y la reindexa al calendario diario continuo.

    Los días que no están en el archivo quedan como filas vacías con estado
    "sin_registro". El caudal faltante (no_data, frozen, spike) no se imputa: queda
    vacío y NeuralHydrology lo excluye de la pérdida y de las métricas."""
    df = pd.read_csv(ruta, sep="\t")
    df.index = pd.to_datetime(df.pop("Fecha"), format="%d/%m/%Y")
    if df.index.duplicated().any():
        raise ValueError(f"Fechas repetidas en {ruta.name}")
    inicio, fin = df.index.min(), df.index.max()

    df = df.rename(columns={**cfg["forzantes"],
                            cfg["objetivo"]["columna"]: "streamflow_m3s",
                            cfg["objetivo"]["estado"]: "streamflow_estado"})
    df = df.reindex(calendario)
    df["streamflow_estado"] = df["streamflow_estado"].fillna("sin_registro")
    return df, inicio, fin


# ----------------------------------------------------------------------- transformación

def rachas(evento: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inicio y fin (exclusivo) de cada racha consecutiva de True."""
    bordes = np.flatnonzero(np.diff(np.r_[0, evento.astype(np.int8), 0]))
    return bordes[::2], bordes[1::2]


def interpolar_huecos_cortos(s: pd.Series, max_dias: int) -> tuple[pd.Series, np.ndarray]:
    """Interpola linealmente solo los huecos internos de hasta `max_dias` días (§4.2).

    Los huecos más largos y los extremos de la serie quedan vacíos. Devuelve la serie
    y una máscara con los días rellenados."""
    rellenar = np.zeros(len(s), dtype=bool)
    for ini, fin in zip(*rachas(s.isna().to_numpy())):
        es_interno = ini > 0 and fin < len(s)
        if es_interno and fin - ini <= max_dias:
            rellenar[ini:fin] = True
    if not rellenar.any():
        return s, rellenar
    interpolada = s.interpolate(method="linear", limit_area="inside")
    return s.where(~rellenar, interpolada), rellenar


def caudal_a_mm(q_m3s: pd.Series, area_km2: float) -> pd.Series:
    """Convierte el caudal de m³/s a lámina de agua sobre la cuenca, en mm/día.

    m³/s × 86 400 s/día × 1 000 mm/m ÷ (área × 10⁶ m²/km²) = m³/s × 86,4 / área.
    Los días sin caudal siguen vacíos."""
    return q_m3s * 86.4 / area_km2


def duracion_media(evento: np.ndarray) -> float:
    """Duración media en días de las rachas consecutivas de True (0 si no hay)."""
    ini, fin = rachas(evento)
    return float((fin - ini).mean()) if len(ini) else 0.0


def indices_climaticos(pr: pd.Series, pet: pd.Series) -> dict[str, float]:
    """Índices climáticos con las definiciones de CAMELS-COL (§4.4).

    Deben recibir SOLO el período de entrenamiento y los valores observados (sin
    interpolar). Los días vacíos cortan las rachas."""
    dias = pr.notna().sum()
    if dias == 0:
        return dict.fromkeys(["p_mean", "pet_mean", "aridity", "high_prec_freq",
                              "high_prec_dur", "low_prec_freq", "low_prec_dur"], np.nan)
    anios = dias / 365.25
    p_mean, pet_mean = pr.mean(), pet.mean()
    alta = (pr >= 5 * p_mean).to_numpy()   # NaN >= x es False: corta la racha
    seca = (pr < 1).to_numpy()
    return {
        "p_mean": p_mean,
        "pet_mean": pet_mean,
        "aridity": pet_mean / p_mean,
        "high_prec_freq": alta.sum() / anios,
        "high_prec_dur": duracion_media(alta),
        "low_prec_freq": seca.sum() / anios,
        "low_prec_dur": duracion_media(seca),
    }


def filtrar_degenerados(attrs: pd.DataFrame, grupos: dict[str, list[str]], cfg_deg: dict) -> list[str]:
    """Atributos de porcentaje casi vacíos: menos de `min_frac_cuencas` de las cuencas
    tienen más de `umbral_pct` % de su área en esa clase."""
    candidatos = [c for g in cfg_deg["grupos"] for c in grupos[g]]
    frac = (attrs[candidatos] > cfg_deg["umbral_pct"]).mean()
    return frac[frac < cfg_deg["min_frac_cuencas"]].index.tolist()


def atributos_del_modelo(attrs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Re-escala el área y elige los atributos estáticos que entran al modelo.

    Los candidatos son los de cfg["atributos_modelo"], donde tc_representative ya
    está reemplazado por tc_Johnstone. Se descartan los atributos casi vacíos."""
    attrs = attrs.assign(log_area=np.log10(attrs["area"]))
    grupos = cfg["atributos_modelo"]
    descartados = filtrar_degenerados(attrs, grupos, cfg["degenerados"])
    modelo = [c for g in grupos.values() for c in g if c not in descartados]

    faltan = [c for c in modelo if c not in attrs.columns]
    if faltan:
        raise ValueError(f"Atributos del modelo que no existen: {faltan}")
    con_nan = attrs[modelo].columns[attrs[modelo].isna().any()].tolist()
    if con_nan:
        raise ValueError(f"Atributos del modelo con vacíos: {con_nan}")
    return attrs, modelo, descartados


def advertir_redundantes(attrs: pd.DataFrame, umbral: float = 0.99) -> None:
    """Avisa de pares de atributos casi idénticos (p. ej. columnas duplicadas)."""
    corr = attrs.corr().abs()
    pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]
             if corr.loc[a, b] > umbral]
    for a, b, r in pares:
        print(f"  AVISO: {a} y {b} tienen |corr| = {r:.3f}")


# ------------------------------------------------------------------------------ main

def asignar_periodo(fechas: pd.DatetimeIndex, periodos: dict) -> pd.Categorical:
    """Etiqueta cada fecha con su período: train, validation o test."""
    etiqueta = np.full(len(fechas), "", dtype=object)
    for nombre in ["train", "validation", "test"]:
        ini, fin = periodos[nombre]
        etiqueta[(fechas >= ini) & (fechas <= fin)] = nombre
    return pd.Categorical(etiqueta, categories=["train", "validation", "test"])


def main(ruta_config: Path) -> None:
    cfg = yaml.safe_load(ruta_config.read_text(encoding="utf-8"))
    rutas = {k: ROOT / v for k, v in cfg["rutas"].items()}
    periodos = {k: [pd.Timestamp(f) for f in v] for k, v in cfg["periodos"].items()}
    calendario = pd.date_range(*periodos["calendario"], freq="D")
    en_train = (calendario >= periodos["train"][0]) & (calendario <= periodos["train"][1])
    forzantes = list(cfg["forzantes"].values())
    max_dias = cfg["interpolacion_max_dias"]

    archivos = sorted(rutas["series"].glob("Hydromet_data_*.txt"))
    ids = [a.stem.removeprefix("Hydromet_data_") for a in archivos]
    print(f"Cuencas encontradas: {len(ids)}")

    attrs = cargar_atributos(rutas["atributos"], rutas["reporte_limpieza"], ids)

    tablas, resumen, climaticos = [], [], {}
    # Días vacíos (dentro del registro de cada estación) e interpolados, por variable
    conteo = {c: {"dias_vacios_en_registro": 0, "dias_interpolados": 0} for c in forzantes + ["streamflow"]}
    periodo = asignar_periodo(calendario, periodos)
    for gid, archivo in zip(ids, archivos):
        df, inicio, fin = leer_serie(archivo, cfg, calendario)
        en_registro = (calendario >= inicio) & (calendario <= fin)

        # Índices climáticos: solo período de entrenamiento y datos observados
        climaticos[gid] = indices_climaticos(df.loc[en_train, "pr"], df.loc[en_train, "poten_evapo"])
        dias_forzantes_train = int(df.loc[en_train, "pr"].notna().sum())

        # Forzantes: interpolar solo huecos cortos
        interpolado = np.zeros(len(df), dtype=bool)
        for col in forzantes:
            conteo[col]["dias_vacios_en_registro"] += int(df.loc[en_registro, col].isna().sum())
            df[col], mascara = interpolar_huecos_cortos(df[col], max_dias)
            conteo[col]["dias_interpolados"] += int(mascara.sum())
            interpolado |= mascara
        df["forzantes_interpolados"] = interpolado

        # Caudal: m³/s -> mm/día. Los vacíos NO se imputan.
        df["streamflow"] = caudal_a_mm(df["streamflow_m3s"], attrs.at[gid, "area"])
        conteo["streamflow"]["dias_vacios_en_registro"] += int(df.loc[en_registro, "streamflow"].isna().sum())

        df = df.rename_axis("date").reset_index()
        df.insert(0, "gauge_id", gid)
        df.insert(2, "periodo", periodo)
        tablas.append(df)

        q_valido = df["streamflow"].notna()
        resumen.append({
            "gauge_id": gid,
            "region": attrs.at[gid, "region"],
            "inicio_registro": inicio.date(),
            "fin_registro": fin.date(),
            "q_valido_pct_registro": round(100 * q_valido[en_registro].mean(), 2),
            **{f"dias_q_{p}": int((q_valido & (df["periodo"] == p)).sum())
               for p in ["train", "validation", "test"]},
            "dias_forzantes_interpolados": int(interpolado.sum()),
            "dias_forzantes_vacios_tras_interpolar": int(df.loc[en_registro, "pr"].isna().sum()),
            "anios_forzantes_train": round(dias_forzantes_train / 365.25, 1),
        })

    # ---- atributos estáticos del modelo
    attrs = attrs.join(pd.DataFrame.from_dict(climaticos, orient="index"))
    attrs, modelo, descartados = atributos_del_modelo(attrs, cfg)
    print(f"Atributos descartados por estar casi vacíos: {descartados}")
    advertir_redundantes(attrs[modelo])

    # ---- tabla única: series + atributos estáticos
    diario = pd.concat(tablas, ignore_index=True)
    diario = diario.merge(attrs[modelo].astype("float32"), left_on="gauge_id",
                          right_index=True, how="left", validate="many_to_one")
    flotantes = diario.select_dtypes("float64").columns
    diario[flotantes] = diario[flotantes].astype("float32")
    diario["gauge_id"] = diario["gauge_id"].astype("category")
    diario["streamflow_estado"] = diario["streamflow_estado"].astype("category")

    # ---- escritura
    salida = rutas["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    diario.to_parquet(salida / "camels_col_diario.parquet", index=False)

    columnas_attrs = cfg["auxiliares"] + modelo + cfg["firmas_hidrologicas"]
    attrs[columnas_attrs].rename_axis("gauge_id").to_csv(salida / "atributos.csv")
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(salida / "resumen_cuencas.csv", index=False)

    interpolacion = pd.DataFrame(conteo).T.rename_axis("variable")
    interpolacion["dias_que_quedan_vacios"] = (interpolacion["dias_vacios_en_registro"]
                                               - interpolacion["dias_interpolados"])
    interpolacion["pct_vacios_interpolados"] = (100 * interpolacion["dias_interpolados"]
                                                / interpolacion["dias_vacios_en_registro"]).round(1)
    interpolacion.to_csv(salida / "resumen_interpolacion.csv")

    finales = {
        "forzantes": forzantes,
        "objetivo": "streamflow",
        "atributos_estaticos": modelo,
        "descartados_por_degenerados": descartados,
        "interpolacion_max_dias": max_dias,
    }
    (salida / "variables_finales.yml").write_text(
        yaml.safe_dump(finales, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # ---- resumen en consola
    print(f"\nTabla diaria: {len(diario):,} filas x {diario.shape[1]} columnas "
          f"({len(ids)} cuencas x {len(calendario):,} días)")
    print(f"Atributos estáticos del modelo: {len(modelo)}")
    print(f"Días con caudal válido: {int(diario['streamflow'].notna().sum()):,}")
    print(f"Días de forzantes interpolados: {int(diario['forzantes_interpolados'].sum()):,} "
          f"(los mismos días en las {len(forzantes)} forzantes; el caudal no se interpola)")
    print(f"Cuencas con caudal en el período de prueba: {(resumen['dias_q_test'] > 0).sum()} de {len(ids)}")
    print(f"\nArchivos escritos en {salida.relative_to(ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "variables.yml")
    main(parser.parse_args().config)
