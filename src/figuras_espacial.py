"""Figuras de la etapa 2 (cuencas no vistas) para el artículo. Uso: python src/figuras_espacial.py

Lee runs/flujo_espacial/series_espacial.parquet y docs/flujo-espacial-cuencas.csv (de resultados_espacial.py) y
escribe en docs/figuras/: espacial-cdf-nse.png, espacial-mapa-nse.png, espacial-hidrogramas.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figuras"
TEXTO, TEXTO2, REJILLA = "#191919", "#666666", "#e4e4e4"
COLORES = {"sim_lstm": "#257de0", "lgbm": "#f47230", "vecino": "#00b894", "clim_donantes": "#9c9c9c"}
NOMBRES = {"sim_lstm": "LSTM regional", "lgbm": "LightGBM regional", "vecino": "Vecino más cercano",
           "clim_donantes": "Climatología de donantes"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXTO2, "axes.labelcolor": TEXTO, "xtick.color": TEXTO2,
                     "ytick.color": TEXTO2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "savefig.dpi": 300})


def cdf(cuencas):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for c, color in COLORES.items():
        v = np.sort(cuencas[f"NSE_{c}"].dropna().clip(lower=-1).to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=2,
                label=f"{NOMBRES[c]} ({np.median(v):.2f})")
    ax.axvline(0.5, color=TEXTO2, lw=0.8, ls="--")
    ax.text(0.52, 0.04, "meta 0,50", color=TEXTO2, fontsize=7)
    ax.set(xlim=(-1, 1), ylim=(0, 1), xlabel="NSE por cuenca (valores < −1 en −1)", ylabel="Fracción de cuencas")
    ax.set_title(f"NSE en las {len(cuencas)} cuencas no vistas (1981–2022)", fontsize=9, loc="left")
    # leyenda debajo del eje para no tapar las curvas
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2,
              title="Mediana entre paréntesis", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "espacial-cdf-nse.png", bbox_inches="tight")


def mapa(cuencas):
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    v = cuencas["NSE_sim_lstm"]
    neg = v < 0
    ax.scatter(cuencas.loc[neg, "lon"], cuencas.loc[neg, "lat"], s=22, marker="x", color=TEXTO2, lw=1.2,
               label="NSE < 0", zorder=3)
    sc = ax.scatter(cuencas.loc[~neg, "lon"], cuencas.loc[~neg, "lat"], c=v[~neg], cmap="Blues", vmin=-0.2, vmax=1,
                    s=26, edgecolor="white", linewidth=0.8, zorder=4)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="NSE del LSTM (cuenca no vista)")
    ax.set(xlabel="Longitud (°)", ylabel="Latitud (°)", aspect="equal")
    ax.set_title("NSE del LSTM por cuenca no vista", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "espacial-mapa-nse.png")


def hidrogramas(series, cuencas):
    """Tres cuencas: la del percentil 75, la mediana y la del percentil 25 del NSE del LSTM, en su año más completo."""
    orden = cuencas["NSE_sim_lstm"].dropna().sort_values()
    elegidas = [orden.index[int(q * (len(orden) - 1))] for q in (0.75, 0.5, 0.25)]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4))
    for ax, g in zip(axes, elegidas):
        df = series[series["gauge_id"] == g].set_index("date")
        anio = df.loc[df["obs"].notna() & df["sim_lstm"].notna()].index.year.value_counts().idxmax()
        d = df.loc[str(anio)]
        ax.plot(d.index, d["obs"], color=TEXTO, lw=1.2, label="Observado")
        ax.plot(d.index, d["sim_lstm"], color=COLORES["sim_lstm"], lw=1.5, label=NOMBRES["sim_lstm"])
        ax.plot(d.index, d["vecino"], color=COLORES["vecino"], lw=1, alpha=0.9, label=NOMBRES["vecino"])
        c = cuencas.loc[g]
        ax.set_title(f"{g} ({c['region']}, {c['area_km2']:,.0f} km²), {anio}: NSE LSTM {c['NSE_sim_lstm']:.2f}, "
                     f"vecino {c['NSE_vecino']:.2f}", fontsize=8, loc="left", color=TEXTO)
        ax.set_ylabel("Caudal (mm/día)")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "espacial-hidrogramas.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cuencas = pd.read_csv(ROOT / "docs" / "flujo-espacial-cuencas.csv", dtype={"gauge_id": str}, index_col="gauge_id")
    series = pd.read_parquet(ROOT / "runs" / "flujo_espacial" / "series_espacial.parquet")
    series["gauge_id"] = series["gauge_id"].astype(str)
    cdf(cuencas)
    mapa(cuencas)
    hidrogramas(series, cuencas)
    print("Figuras en", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
