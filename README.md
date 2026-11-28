# Proyecto-final-ADL

Predicción del caudal medio diario en 248 cuencas de Colombia (CAMELS-COL) con una LSTM regional (NeuralHydrology).
**Regla del proyecto:** el modelo no recibe caudal observado como entrada, ni nada derivado de él.

## Estado

**Flujo temporal (20/11):** prueba 2016–2022, 183 cuencas, en los mismos días. Valores medianos entre cuencas.

| Modelo | NSE | KGE |
|---|---|---|
| LSTM con 4 forzantes y 48 atributos | 0,394 | 0,513 |
| LSTM + LightGBM | 0,410 | 0,518 |
| LSTM con ERA5-Land, ensamble de 3 semillas | 0,432 | 0,539 |
| **Modelo final: LSTM + LightGBM, los dos con ERA5-Land** | **0,437** | **0,536** |
| Meta de la propuesta | 0,70 | 0,70 |

**Flujo espacial (28/11):** cuencas sin estación, 95 cuencas retenidas por sistema de ríos en 5 particiones, 1981–2022. Valores medianos entre cuencas.

| Modelo | NSE | KGE |
|---|---|---|
| Climatología de 5 donantes | −0,081 | −0,058 |
| Vecino más cercano (caudal observado de otra estación) | −0,426 | 0,231 |
| IDW de las 5 estaciones más cercanas (caudal observado) | −0,114 | 0,288 |
| LightGBM regional | 0,257 | 0,445 |
| LSTM con 4 forzantes y 48 atributos, ensamble de 3 semillas | 0,276 | 0,438 |
| **LSTM con ERA5-Land, ensamble de 3 semillas** | **0,300** | **0,453** |
| Meta de la propuesta | 0,50 | — |

Detalle en la bitácora (sección 24), en el notebook de modelado (secciones 7, 8 y 10) y en `docs/plan-accion-cuencas-sin-estacion.md`. Para reproducirlo: `python src/flujo_espacial.py entrenar --variante era5l`, `python src/baselines_espacial.py`, `python src/resultados_espacial.py --variante era5l` y `python src/figuras_espacial.py --variante era5l`.

## Qué hay en el repositorio

| Archivo o carpeta | Contenido |
|---|---|
| `proyecto-final-preprocesamiento.ipynb` | **Parte 1.** Sección 1: CAMELS-COL limpio en `data/processed/`. Sección 2: 15 forzantes de ERA5-Land y una segunda base en `data/processed_era5land/` |
| `proyecto-final-modelado.ipynb` | **Parte 2.** Configuración, estrategia, particiones espaciales, etapa 1 (flujo temporal: LSTM, LightGBM, ERA5-Land, selección de variables y modelo final), etapa 2 (flujo espacial), baselines y resultados de cada etapa |
| `docs/bitacora-modelado.md` | **Registro de cada paso, decisión y resultado.** Conviene empezar por el resumen del inicio |
| `docs/registro-variables.csv` | Variables usadas en cada iteración, con su fuente, su cálculo y su efecto medido |
| `docs/variables-candidatas.csv`, `docs/alternativas-mejora.csv` | Catálogos de variables y de alternativas para mejorar el modelo |
| `docs/diagnostico-cuencas.csv` | Diagnóstico por cuenca: por qué falla el modelo |
| `configs/` | `base.yml`, listas de cuencas (`basins/`) y la configuración exacta de cada corrida (`corridas/`) |
| `src/download.py` | Descarga CAMELS-COL desde Zenodo |
| `src/*_espacial.py` | Etapa 2: entrenamiento por partición, referencias, resultados y figuras |
| `docs/plan-accion-cuencas-sin-estacion.md` | Plan de acción para la pregunta de cuencas sin estación y la entrega final |

`data/` y `runs/` no están en git porque pesan varios GB. Los resultados se ven en las salidas guardadas de los notebooks y en la bitácora.

## Cómo reproducir

1. Instalar PyTorch con CUDA, luego `pip install -r requirements.txt` y `pip install neuralhydrology`.
2. Ejecutar los notebooks desde la raíz del repositorio, primero el de preprocesamiento y después el de modelado.
   - El preprocesamiento parte de las series ya filtradas en `data/datos-originales/` (ver su sección 1.2).
   - ERA5-Land (sección 2.1) requiere una cuenta de Copernicus CDS y la clave en `~/.cdsapirc`, nunca en el repositorio. La descarga tardó unas 31 horas por la cola del CDS y ocupa unos 11 GB.
3. Las celdas que entrenan, descargan o reconstruyen datos tienen una bandera `EJECUTAR_* = False`. Con la bandera así, la celda usa lo que ya está en `runs/` o en `data/`, y si no existe, lo indica.
