# Proyecto-final-ADL

Predicción del caudal medio diario en 248 cuencas de Colombia (CAMELS-COL) con una LSTM regional (NeuralHydrology).
**Regla del proyecto:** el modelo no recibe caudal observado como entrada, ni nada derivado de él.

## Estado (20/11/2026)

Flujo temporal: prueba 2016–2022, 183 cuencas, en los mismos días. Valores medianos entre cuencas.

| Modelo | NSE | KGE |
|---|---|---|
| LSTM con 4 forzantes y 48 atributos | 0,394 | 0,513 |
| LSTM + LightGBM | 0,410 | 0,518 |
| LSTM con ERA5-Land, ensamble de 3 semillas | 0,432 | 0,539 |
| **Modelo final: LSTM + LightGBM, los dos con ERA5-Land** | **0,437** | **0,536** |
| Meta de la propuesta | 0,70 | 0,70 |

El flujo espacial (cuencas no aforadas, 5 particiones) está implementado pero no se ha entrenado.

## Qué hay en el repositorio

| Archivo o carpeta | Contenido |
|---|---|
| `proyecto-final-preprocesamiento.ipynb` | **Parte 1.** Sección 1: CAMELS-COL limpio en `data/processed/`. Sección 2: 15 forzantes de ERA5-Land y una segunda base en `data/processed_era5land/` |
| `proyecto-final-modelado.ipynb` | **Parte 2.** Configuración, prueba de humo, particiones, entrenamiento y evaluación. Las secciones 8.4 a 8.7 tienen la selección de variables, la confirmación con el LSTM y la evaluación final |
| `docs/bitacora-modelado.md` | **Registro de cada paso, decisión y resultado.** Conviene empezar por el resumen del inicio |
| `docs/registro-variables.csv` | Variables usadas en cada iteración, con su fuente, su cálculo y su efecto medido |
| `docs/variables-candidatas.csv`, `docs/alternativas-mejora.csv` | Catálogos de variables y de alternativas para mejorar el modelo |
| `docs/diagnostico-cuencas.csv` | Diagnóstico por cuenca: por qué falla el modelo |
| `configs/` | `base.yml`, listas de cuencas (`basins/`) y la configuración exacta de cada corrida (`corridas/`) |
| `src/download.py` | Descarga CAMELS-COL desde Zenodo |

`data/` y `runs/` no están en git porque pesan varios GB. Los resultados se ven en las salidas guardadas de los notebooks y en la bitácora.

## Cómo reproducir

1. Instalar PyTorch con CUDA, luego `pip install -r requirements.txt` y `pip install neuralhydrology`.
2. Ejecutar los notebooks desde la raíz del repositorio, primero el de preprocesamiento y después el de modelado.
   - El preprocesamiento parte de las series ya filtradas en `data/datos-originales/` (ver su sección 1.2).
   - ERA5-Land (sección 2.1) requiere una cuenta de Copernicus CDS y la clave en `~/.cdsapirc`, nunca en el repositorio. La descarga tardó unas 31 horas por la cola del CDS y ocupa unos 11 GB.
3. Las celdas que entrenan, descargan o reconstruyen datos tienen una bandera `EJECUTAR_* = False`. Con la bandera así, la celda usa lo que ya está en `runs/` o en `data/`, y si no existe, lo indica.
