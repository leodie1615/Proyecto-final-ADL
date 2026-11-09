# CAMELS-COL — Handoff de implementación

> Documento para retomar el proyecto en VS Code / Claude Code y pasar a la fase de implementación.
> Estado al **9 de noviembre de 2026**. La propuesta (Entrega 1) ya fue entregada y aprobada en diseño.

---

## 1. Qué se va a construir

**Un modelo LSTM regional único** que simula **caudal medio diario** para las 346 cuencas de CAMELS-COL a partir de forzantes meteorológicas y atributos estáticos de cuenca.

**Autores:** Santiago Palacios Loaiza, Diego Leonardo Barón Espitia, Andrés Felipe Mosquera Hernández, Juan Sebastián Rodríguez Avella — Universidad de los Andes.

### Regla no negociable

> **NO se usa caudal rezagado como entrada.** Ni `Q(t-1)`, ni medias móviles del caudal, ni nada derivado del caudal observado de la cuenca objetivo.

Es **simulación**, no pronóstico. El modelo recibe meteorología y devuelve caudal. Esto es lo que permite evaluarlo en cuencas no aforadas y, eventualmente, simular escenarios climáticos.

### Metas comprometidas

| Objetivo | Meta |
|---|---|
| Pipeline | 346 cuencas procesadas y listas |
| Modelo regional | **NSE y KGE medianos ≥ 0,70** en período de prueba |
| Generalización espacial | **NSE mediano ≥ 0,50** en cuencas no vistas |

---

## 2. Entorno y presupuesto de cómputo

**Hardware disponible:** laptop con NVIDIA RTX 3050 6 GB, Intel Core i5-12450H, 38 GB RAM. Google Colab como alternativa.

**Benchmark medido:** 21,5 ms por actualización de parámetros, 0,3 GB de memoria GPU.

### ⚠️ Planificación de tiempo — revisar antes de arrancar

| Concepto | Cálculo | Tiempo |
|---|---|---|
| Muestras de entrenamiento | ~3 millones | — |
| Una época | 3M / 64 × 21,5 ms | ~17 min |
| Un entrenamiento (30 épocas) | | **~8,5 h** |
| Modelo principal | 1 entrenamiento | 8,5 h |
| Validación cruzada espacial | 5 particiones | **42,5 h** |
| **Total mínimo** | | **~51 h** |

Y eso **sin contar** el ajuste del tamaño de capa oculta ni reintentos por bugs.

**Implicaciones prácticas:**
- Entrenar las 5 particiones **en serie durante varios días**, o paralelizar en Colab con varias cuentas.
- **Checkpointing obligatorio.** Un corte de luz a las 7 horas no puede costar el día.
- Probar todo el pipeline primero con **2–3 épocas y 20 cuencas** antes de lanzar la corrida real.
- Colab gratis desconecta por inactividad; para corridas largas conviene el runtime local o Colab Pro.

---

## 3. Datos

**Fuente:** CAMELS-COL — https://zenodo.org/records/18794895
DOI: `10.5281/zenodo.18794895`
Paper: https://essd.copernicus.org/preprints/essd-2025-200/

**Descargar solo** el archivo de series hidrometeorológicas (~184,6 MB) y los `.xlsx` de atributos. **NO descargar el ráster de 3,2 GB**, no se usa.

### Cobertura

346 cuencas, 1981–2022 (42 años).

| Región | Estaciones |
|---|---|
| Magdalena-Cauca | 182 |
| Orinoco | 57 |
| Caribe | 53 |
| Pacífico | 32 |
| Amazonas | 22 |

*(Nota: el paper reporta 347 y algunos conteos regionales difieren en ±2. Si se recontó sobre los archivos descargados, dejar nota al pie explicándolo.)*

### Variables dinámicas (forzantes del modelo)

| Columna | Descripción | Unidad | Fuente |
|---|---|---|---|
| `pr` | Precipitación | mm/día | CHIRPS v2 |
| `poten_evapo` | Evapotranspiración potencial | mm/día | MSWX (Hargreaves-Samani) |
| `t_max` | Temperatura máxima | °C | MSWX |
| `t_min` | Temperatura mínima | °C | MSWX |
| `t_mean` | Temperatura media | °C | MSWX |
| **`streamflow`** | **Caudal medio diario — OBJETIVO** | **m³/s** | IDEAM |

*La propuesta menciona 4 forzantes (pr, ETo, t_min, t_max). Decidir si se incluye `t_mean` — es combinación lineal de las otras dos, probablemente redundante.*

### Atributos estáticos (~71)

**SE USAN:** fisiográficos (`area`, `perimeter`, `gravelius_index`, `factor_form`, `streng_chanel`, `equi_slope`, `cn_catchment`, 4 tiempos de concentración), índices climáticos (recalculados, ver §4), cobertura MapBiomas (6), geología (7 litologías), suelos IGAC (17).

**SE EXCLUYEN — las 11 firmas hidrológicas:**
```
q_mean, runoff_ratio, stream_elas, slope_fdc, baseflow_index,
Q5, Q95, high_q_freq, high_q_dur, low_q_freq, low_q_dur
```
Se calculan a partir del caudal observado → **no existen en una cuenca no aforada** y causarían fuga de información.

---

## 4. Pipeline de preparación — especificación

Ejecutar en este orden.

### 4.1 Reindexado temporal
Los días sin registro **no aparecen como filas** en los archivos originales. Reindexar cada serie a un calendario diario continuo 1981-01-01 → 2022-12-31.

```python
idx = pd.date_range("1981-01-01", "2022-12-31", freq="D")
df = df.reindex(idx)
```

### 4.2 Vacíos
- **Forzantes:** interpolación lineal.
- **Caudal:** **NO imputar.** Se excluye del cálculo de la pérdida y de las métricas (máscara de NaN).
- **No se descarta ninguna cuenca por datos faltantes.** Un umbral del 5% eliminaría cerca de la mitad del conjunto, y los autores del dataset ya exigieron 70% de disponibilidad.

### 4.3 Conversión de unidades
```python
# m³/s → mm/día
Q_mm = Q_m3s * 86400 / (area_km2 * 1000)
```
Obligatorio. Sin esto, las cuencas grandes dominan la pérdida.

### 4.4 Recálculo de índices climáticos
`aridity`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur` vienen calculados sobre **todo el registro** en el dataset original. Recalcularlos **solo con los forzantes del período de entrenamiento** para que no incorporen información de los años de prueba.

Definiciones del paper:
- `aridity` = ETo anual total / precipitación anual total
- `high_prec_freq` = días/año con Pr ≥ 5 × precipitación media diaria
- `high_prec_dur` = duración media de eventos de alta precipitación (días consecutivos)
- `low_prec_freq` = días/año con Pr < 1 mm/día
- `low_prec_dur` = duración media de períodos secos

### 4.5 Limpieza de atributos degenerados
Revisar y descartar columnas que son cero (o casi) en casi todas las cuencas — típicamente `spodosols_perc`, `coal_mine_pit_perc`, `perpetual_snow_perc`. Aportan ruido.

### 4.6 Estandarización
**Z-score usando ÚNICAMENTE estadísticos de las cuencas de entrenamiento y del período de entrenamiento.**

```python
mu = X[cuencas_train, periodo_train].mean()
sd = X[cuencas_train, periodo_train].std()
# estos mismos mu/sd se aplican a TODO lo demás
```

Aplica a forzantes, atributos **y** caudal. En validación cruzada, recalcular por partición.

> ⚠️ **Fuga por normalización.** Es el error más sutil y el más común. Si se calculan sobre todo el dataset, se filtra información del test al modelo.

### 4.7 Partición temporal
| Período | Años |
|---|---|
| Entrenamiento | 1981–2010 |
| Validación | 2011–2015 |
| Prueba | 2016–2022 |

Justificación (Yu & Tolson 2026): los datos recientes son los más informativos, por eso el período de prueba son los años más recientes.

**Nota:** solo **252 de las 346 cuencas** tienen registros en el período de prueba. → Verificar si las 94 restantes se concentran en alguna región; si sí, el resultado temporal queda sesgado y hay que declararlo.

### 4.8 Ventanas
Secuencias de **365 días** de forzantes + vector de atributos estáticos → predicción del caudal del **último día** de la ventana.

---

## 5. Modelo

### Configuración inicial (adoptada de Wi et al. 2025 y Alzhanov et al. 2025)

```yaml
model: cudalstm
hidden_size: 256          # único hiperparámetro a ajustar
output_dropout: 0.4
optimizer: Adam
learning_rate: 5e-4
batch_size: 64
seq_length: 365
epochs: 30
loss: NSE                 # NSE* normalizado por cuenca
```

### Función de pérdida: NSE*

NSE normalizado por la desviación estándar del caudal **de cada cuenca**, calculada **solo con el período de entrenamiento**. Evita que las cuencas grandes dominen el entrenamiento.

En NeuralHydrology es `loss: NSE` (ya implementa la versión de Kratzert con el término de estabilidad).

### Ajuste de hiperparámetros
Solo se ajusta `hidden_size`. Se usa el período de validación de las **216 cuencas que nunca se retienen** en la validación cruzada espacial (Magdalena-Cauca 176 + Orinoco 40). Parada temprana sobre ese mismo conjunto.

Justificación: la complejidad del LSTM debe corresponder al número de cuencas de entrenamiento (He et al. 2024).

### Librería
**NeuralHydrology** — https://github.com/neuralhydrology/neuralhydrology

> ⚠️ **CAMELS-COL no es un dataset soportado nativamente.** Hay dos caminos:
> 1. Usar `dataset: generic` (GenericDataset), que lee series por cuenca en `time_series/*.nc` y atributos en `attributes/*.csv`. **Requiere convertir CAMELS-COL a ese formato.** → *Verificar la especificación exacta en la documentación antes de escribir el conversor.*
> 2. Escribir una clase `CamelsCOL(BaseDataset)` propia.
>
> La opción 1 es más rápida. **Esta conversión es probablemente la primera tarea de código del proyecto.**

---

## 6. Diseño de validación

### 6.1 Validación temporal
Cuencas de entrenamiento, período 2016–2022 (nunca visto). Mide: *¿el modelo aprendió hidrología o memorizó las series?*

### 6.2 Validación espacial — 5 particiones por sistemas de ríos

> **Este es el aporte metodológico más fuerte del proyecto. No simplificarlo.**

**El problema:** 324 de las 346 cuencas están **anidadas** (tienen otra cuenca aguas arriba o aguas abajo). Retener una cuenca y entrenar con su cuenca anidada **filtra su caudal al modelo** — el caudal de la cuenca aguas arriba es literalmente parte del de la de abajo.

**La solución:** las particiones se forman con **sistemas de ríos completos**, es decir, grupos de cuencas conectadas por anidamiento.

| Grupo | Cuencas | Rol |
|---|---|---|
| Sistema Magdalena-Cauca | 176 | **Siempre en entrenamiento** |
| Sistema Orinoco | 40 | **Siempre en entrenamiento** |
| 44 sistemas restantes | 130 | Repartidos en **5 particiones de 26** |

En cada iteración: entrenar con 320 cuencas, evaluar sobre las 26 retenidas. Las retenidas funcionan como no aforadas — el modelo nunca vio su caudal **ni el de ninguna cuenca conectada con ellas**.

**Por partición:** la estandarización y la parada temprana usan solo las cuencas de entrenamiento de esa partición.

**Evaluación:** como el modelo no vio las cuencas retenidas en ningún año, se evalúan **sobre todo su registro**.

**⚠️ Sesgo conocido y declarado:** las 130 cuencas evaluadas cubren **todo** el Caribe, el Pacífico y el Amazonas, pero solo 17 del Orinoco y 6 de Magdalena-Cauca. Son justamente las regiones más atípicas (Pacífico ultra-húmedo, Caribe seco, Amazonía con red débil). → **Enmarcar la meta de 0,50 como cota inferior conservadora**, no como estimación del desempeño típico.

**Tarea de código pendiente:** identificar los sistemas de ríos a partir de los shapefiles de límites de cuenca. Dos cuencas están anidadas si el polígono de una contiene el punto de aforo de la otra. Esto hay que calcularlo — no viene dado en el dataset.

---

## 7. Métricas

Calculadas **por cuenca**, tras **desnormalizar** las predicciones, y resumidas con la **mediana** entre cuencas.

**Comprometidas:** NSE, KGE, RMSE, MAE.

**Recomendadas adicionalmente** (quince líneas de código sobre predicciones que ya van a existir):
- **FHV** — sesgo % en el 2% superior de la curva de duración → caudales altos
- **FLV** — sesgo % en el 30% inferior → caudales bajos
- **FMS** — sesgo en la pendiente del tramo medio

Referencia: Yilmaz, Gupta & Wagener (2008), WRR 44, W09417.

**Contexto:** los tres papers de referencia reportan **FHV negativo en todos sus modelos** (−12 a −45), es decir, todos subestiman los picos. Esperen lo mismo.

**Por qué conviene incluirlas:** el profesor de los Andes había pedido explícitamente foco en caudales altos y bajos, y la propuesta actual no lo cubre. Recuperarlo es barato.

### Presentación de resultados
**Distribuciones, no promedios.** Boxplot o CDF del NSE por régimen de evaluación. Se ve de un vistazo la mediana **y** cuántas cuencas quedaron bajo NSE cero.

> **Recordatorio:** el NSE no tiene límite inferior. Una sola cuenca con NSE −220 destruye el promedio. **Siempre mediana.**

---

## 8. Hueco a llenar: baseline

**No hay baseline en la propuesta.** El objetivo 2 fija NSE ≥ 0,70 sin referente. Un jurado va a preguntar si 0,70 es bueno.

Opciones baratas, en orden de esfuerzo:
1. **Climatología diaria** — predecir, para cada cuenca y cada día del año, la media histórica de ese día. Trivial de implementar.
2. **Regresión lineal con rezagos de precipitación** (1–10 días) + atributos.
3. **LightGBM** sobre variables agregadas (precipitación acumulada a 1/3/7/30 días, ETo, temperatura, atributos).

Con que la LSTM le saque 0,20–0,30 de NSE, el resultado se defiende solo.

---

## 9. Errores que invalidan el experimento

1. **Fuga por normalización** — estadísticos calculados sobre todo el dataset (§4.6).
2. **Split aleatorio de días** — con autocorrelación fuerte el NSE se dispara sin significar nada. **Siempre cronológico.**
3. **Usar el período de prueba para parada temprana** — para eso está validación.
4. **Imputar el caudal faltante** — entrena sobre valores sintéticos. Enmascarar.
5. **Ignorar el anidamiento** al formar particiones espaciales (§6.2).
6. **Usar las firmas hidrológicas como entrada** (§3).
7. **Olvidar recalcular los índices climáticos** con el período de entrenamiento (§4.4).

---

## 10. Estructura de repositorio sugerida

```
camels-col-lstm/
├── data/
│   ├── raw/                  # descarga cruda de Zenodo (gitignored)
│   ├── interim/              # series reindexadas, unidades convertidas
│   └── processed/            # formato NeuralHydrology
│       ├── time_series/      # un .nc por cuenca
│       └── attributes/       # .csv de atributos
├── configs/
│   ├── base.yml              # configuración del modelo
│   └── folds/fold_{1..5}.yml
├── src/
│   ├── download.py
│   ├── preprocess.py         # §4 completo
│   ├── nesting.py            # identificación de sistemas de ríos (§6.2)
│   ├── folds.py              # construcción de las 5 particiones
│   ├── baseline.py           # §8
│   ├── metrics.py            # NSE, KGE, RMSE, MAE, FHV, FLV, FMS
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_resultados.ipynb
├── runs/                     # checkpoints y logs (gitignored)
├── reports/figures/
├── environment.yml
└── README.md
```

---

## 11. Orden de trabajo sugerido

1. **EDA real** — contar vacíos por cuenca y región, verificar qué cuencas tienen datos en 2016–2022, confirmar que `pr` está presente y revisar las banderas de control de calidad.
2. **Identificación de anidamiento** (`nesting.py`) — es la pieza más incierta y bloquea las particiones. Atacarla temprano.
3. **Conversor a formato NeuralHydrology** — verificar primero la especificación de `GenericDataset` en la documentación.
4. **Pipeline de preprocesamiento** completo (§4).
5. **Prueba de humo:** 20 cuencas, 2 épocas. Verificar que entrena, que la pérdida baja y que las métricas se calculan.
6. **Baseline** (§8).
7. **Ajuste de `hidden_size`** sobre las 216 cuencas fijas.
8. **Entrenamiento del modelo principal** (~8,5 h).
9. **Validación cruzada espacial** (5 × 8,5 h ≈ 42,5 h).
10. **Análisis y figuras.**

---

## 12. Preguntas abiertas

- ¿Se incluye `t_mean` como forzante o es redundante con `t_max`/`t_min`?
- ¿Las 94 cuencas sin datos en 2016–2022 se concentran en alguna región?
- ¿Cómo se identifica el anidamiento exactamente — contención de polígonos, o hay algún identificador en el dataset?
- ¿Se agregan FHV/FLV/FMS? (recomendado: sí)
- ¿Se agrega baseline? (recomendado: sí)
- ¿Dónde se entrena finalmente — laptop local o Colab? Afecta la estrategia de checkpointing.

---

## 13. Referencias clave para la implementación

| Ref | Uso en este proyecto |
|---|---|
| **Jimenez et al. (2026)** — CAMELS-COL, Zenodo `10.5281/zenodo.18794895` | El dataset |
| **Chao et al. (2026)** — Mixture-of-Experts, Sierra Nevada, *Hydrology Research* | Ventana de 365 días, concatenación forzantes+atributos, modelo compartido generaliza mejor |
| **Wi et al. (2025)** — JoH 661, 133764 | Hiperparámetros base, conversión a mm/día, mediana entre cuencas |
| **Barbhuiya & Gupta (2025)** — EMS 194, 106696 | Exclusión del caudal faltante de la pérdida; partición espacial estricta |
| **He et al. (2024)** — JoH 639, 131638 | Modelo regional > especialización por clases cuando hay pocas cuencas por clase; tamaño del modelo vs. número de cuencas |
| **Yu & Tolson (2026)** — JoH 668, 135046 | Datos recientes son más informativos → período de prueba al final |
| **Alzhanov et al. (2025)** — EMS 194, 106691 | Transfer learning desde Caravan (posible extensión futura) |
| **Kratzert et al. (2022)** — JOSS 7(71), 4050 | NeuralHydrology |
| **Yilmaz et al. (2008)** — WRR 44, W09417 | FHV / FLV / FMS |

---

## 14. Contexto de negocio (para el informe final)

- ~**70%** de la producción eléctrica colombiana proviene de centrales hidráulicas.
- El área **Magdalena-Cauca concentra 74% de la población y 80% del PIB**, pero tiene la menor oferta hídrica del país.
- **Bogotá enfrentó su primer racionamiento en cuatro décadas** durante El Niño de 2024.
- La **ola invernal de La Niña 2010–2011** afectó a 5,2 millones de personas y costó cerca del **2% del PIB**.
- El IDEAM estima en ~100% la probabilidad de que El Niño persista hasta febrero de 2027, y en 75% la de que sea el más fuerte registrado *(verificar redacción exacta del comunicado antes de citarlo)*.
- De **921 estaciones del IDEAM**, solo 346 pasaron los filtros de calidad → buena parte del territorio no tiene medición confiable.

**Aplicaciones:** operación de embalses, despacho energético, alertas de creciente en cuencas sin estaciones.
