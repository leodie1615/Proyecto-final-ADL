# Plan de acción: ¿cómo predecir el caudal medio diario en Colombia en cuencas sin estación de medición con deep learning?

27/11/2026, actualizado el 28/11 con la rúbrica de la entrega final (`01-enunciado/entrega-final/entrega-final.md`; sección 10). Basado en los 67 documentos de `05-referencias/md/` (con su PDF), los 19 repositorios de `05-referencias/repos/`, la bitácora (`docs/bitacora-modelado.md`) y el estado del repositorio. Fechas que mandan: presentación el **2/12 a las 5:30 pm** y artículo el **4/12**.

---

## 1. La respuesta corta, y qué falta para poder darla con datos

**Cómo se hace (consenso de la literatura).** Con **un solo LSTM regional** entrenado con **muchas cuencas aforadas a la vez**. Recibe, día a día, el clima de la cuenca (lluvia, evapotranspiración y temperaturas) y sus **atributos estáticos** (relieve, clima medio, suelos, cobertura, geología). **Nunca recibe caudal**, ni rezagado ni de donantes. Como todo lo que recibe existe en cualquier polígono, se puede aplicar a una cuenca sin estación. Su capacidad se mide con **validación cruzada espacial**: se retienen cuencas enteras, junto con su sistema de ríos, y se simulan sin haber visto nunca su caudal.

| Evidencia | Qué muestra |
|---|---|
| Kratzert et al. 2019 (WRR), `kratzert__lstm_for_pub` | En 531 cuencas de EE. UU. con *k-fold* espacial, el LSTM sin caudal de la cuenca (NSE mediano 0,69) supera a SAC-SMA **calibrado en la propia cuenca** (0,64) y al National Water Model (0,58). El repositorio usa 12 particiones y 10 reinicios |
| Arsenault et al. 2023 (HESS) | En 148 cuencas, el LSTM supera a la regionalización con modelos conceptuales en el 93–97 % de las cuencas |
| Barbhuiya y Gupta 2025 | En 144 cuencas de India, el UBLSTM, sin caudal rezagado, logra NSE 0,69 en cuencas pseudo no aforadas. Los Transformers rinden peor. Más cuencas de entrenamiento mejoran la generalización |
| Kratzert et al. 2024 (HESS) | Nunca entrenar un LSTM con una sola cuenca: siempre con cientos |
| Lee y Kim 2025 | En PUB (671 cuencas), el LSTM regionalizado es el más preciso frente a HBV regionalizado y a un híbrido dPL-HBV. Su falla típica es subestimar los picos |
| Yu y Tolson 2026 | En PUB, sumar cuencas mejora mucho más que en el flujo temporal, sobre todo con datos recientes. Si hay que recortar, mejor quitar años viejos que cuencas |

**Lo que el proyecto ya tiene.** Todo lo anterior está implementado: 248 cuencas limpias, 35 sistemas de ríos, 5 particiones de 19 cuencas sin fuga por anidamiento y la regla de no usar caudal como entrada.

**Lo que faltaba (hasta el 27/11):** **el flujo espacial nunca se había entrenado.** Sin él, el proyecto responde "¿qué tan bien simula años nuevos en cuencas aforadas?" (NSE 0,437), pero no la pregunta del título.

**Respuesta medida (28/11, 3 semillas, 95 cuencas no vistas, 1981–2022):** el LSTM regional logra un **NSE mediano de 0,276** (KGE 0,438). Supera a la climatología de 5 donantes (−0,081; mejor en 79 de 95 cuencas) y al vecino más cercano con caudal observado (−0,426; mejor en 83 de 95), las dos con p ≈ 0. Empata con LightGBM regional (0,257; p = 0,997). No vista frente a vista, en las mismas 62 cuencas de 2016–2022: 0,248 frente a 0,425. Por región: Orinoco 0,64, Caribe 0,26, Amazonía 0,23, Pacífico 0,15, Magdalena-Cauca −0,03. Queda lejos de la meta de 0,50, y ese es el resultado que se reporta (sección 5). **Con ERA5-Land (B1, 3 semillas, 28/11):** NSE mediano 0,300 frente a 0,276 del base; mejora en 54 de 95 cuencas, pero sin significancia (p = 0,08). La brecha vista − no vista es casi igual (−0,14 en las dos). El preliminar de 2 semillas (0,248, p = 0,79) estaba arrastrado por la semilla 43 y no se sostuvo.

---

## 2. Qué dice la evidencia, por familia de método, y qué hacer con ella

Leyenda: ✅ hacer (barato y con evidencia), 🟡 hacer si alcanza el tiempo, 🔵 trabajo futuro, ⛔ choca con la regla del proyecto (sin caudal como entrada) y solo vale como referencia.

### 2.1 LSTM regional y atributos estáticos

| Fuente | Hallazgo | Implicación para nosotros |
|---|---|---|
| Kratzert et al. 2019 (HESS), EA-LSTM; `kratzert__ealstm_regional_modeling` | Un modelo "universal" aprende comportamientos regionales y locales. La EA-LSTM iguala al LSTM y aporta interpretabilidad (embeddings de cuencas) | ✅ Seguimos con `cudalstm`. 🟡 Una EA-LSTM (`ealstm` en NeuralHydrology) serviría para mostrar en qué cuencas de entrenamiento se "apoya" cada cuenca no vista |
| *Better data or architecture?* 2026 (EGUsphere) | En PUB, **la calidad de los atributos estáticos pesa más que la arquitectura**. Los embeddings de AlphaEarth fueron lo mejor, pero solo junto con agregados del DEM, coordenadas y meteorología. Un EMB-LSTM simple rinde igual que uno con atención | 🟡 **La palanca más prometedora para la etapa 2.** Ya está diseñada en el notebook (sección 11): AlphaEarth reducido con PCA por partición. Agregar también agregados de las forzantes (estacionalidad de la lluvia, bimodalidad), que valen igual en cuencas sin estación |
| Yu et al. 2024 (WRR); He et al. 2024; Wang et al. 2026 | Clasificar cuencas y entrenar un LSTM por grupo ayuda poco (Wang: unos +0,005 de mediana) y **empeora cuando el grupo tiene pocas cuencas** (He). Los atributos estáticos ya hacen el trabajo de la clasificación (Yu) | ⛔ No dividir el modelo por región: Amazonas (11) y Pacífico (20) quedarían con muy pocas cuencas. Sí reportar el desempeño por región |
| Feng et al. 2020 | En regiones enteras retenidas (PUR), el LSTM base da KGE 0,56. Con **curvas de duración** de la zona sube a 0,62, y un ensamble de modelos con distintas entradas evita resultados catastróficos | ✅ Ensamble de semillas (en curso). 🔵 Las curvas de duración regionalizadas de Colombia (Antioquia, Santander) podrían ser una entrada "blanda" que no usa caudal de la cuenca |
| Gauch et al. 2021, *The proper care and feeding of CAMELS* (`gauchm__ealstm_regional_modeling`) | Con pocos datos, árboles y LSTM rinden parecido; con más datos gana el LSTM | ✅ Por eso LightGBM entra como referencia en la etapa 2 |
| Nogueira Filho et al. 2022 (Water), semiárido de Ceará; `francisconog__RegBaciasCE` | Como método de regionalización, todas las redes (LSTM y FFNN) superan al modelo conceptual SMAP regionalizado. Pero el LSTM no rinde mejor que una FFNN simple (NSE medianos de 0,66 a 0,87 según la configuración), así que con pocas cuencas su complejidad no se justifica | ✅ Refuerza dos decisiones: comparar contra modelos más simples (LightGBM, sección 2.7) y entrenar con todas las cuencas, porque la ventaja del LSTM aparece con volumen de datos (Gauch 2021) |
| Chao et al. 2026 | Mezcla de expertos (MoE-τ): generaliza mejor que el LSTM a cuencas retenidas. Ventana de 1 año, elevación y uso del suelo como entradas clave | 🔵 Requiere un modelo propio en NeuralHydrology |

### 2.2 Híbridos diferenciables (física + deep learning)

| Fuente | Hallazgo | Implicación |
|---|---|---|
| Feng et al. 2023 (HESS) | En PUB, los modelos δ se acercan al LSTM o lo superan. **En PUR (regiones retenidas, extrapolación) superan al LSTM**, también en tendencias decadales | 🟡 Nuestras particiones se parecen más a PUR: todo el Pacífico, el Caribe y el Amazonas está en las particiones, y el entrenamiento lo domina Magdalena-Cauca. Es la familia con mejor evidencia justo para nuestro caso. NeuralHydrology 1.13 trae `hybrid_model` |
| Feng et al. 2024 (δHBV-globe) | En 3753 cuencas, la generalización espacial del δHBV (KGE 0,64) es mejor que la del LSTM y que la regionalización clásica | Igual que arriba |
| Funato et al. 2025 | Con solo un 20 % de cuencas aforadas, el LSTM se desploma (NSE −0,61) y un ensamble de modelos conceptuales con *reservoir computing* aguanta (0,51) | ⚠️ Riesgo para particiones con pocas cuencas parecidas en el entrenamiento: medir el NSE de cada partición por separado |
| Lee y Kim 2025; Huynh et al. 2023 (`DassHydro__smash`, `nghi-truyen__Regionalization-Learning`) | El híbrido no supera al LSTM en PUB, pero sirve para diagnosticar fallas. HDA-PR llega a NSE 0,60–0,71 en extrapolación de aguas arriba a aguas abajo | 🔵 |
| Kapoor y Chandra 2026 (QDeepGR4J; `DARE-ML__DeepGR4J-Extremes`), citado en el artículo | Híbrido GR4J + red neuronal con regresión por cuantiles (5 %, 50 % y 95 %) y ensamble: mejora los caudales altos y da un intervalo de incertidumbre. Es un trabajo temporal, en cuencas aforadas de CAMELS-AUS | 🔵 No responde PUB. Su idea útil es **la banda de incertidumbre por cuantiles**, que serviría para el despliegue (fase C, punto 4) en lugar de la banda por región. Trabajo futuro |
| Kratzert et al. 2024 | Hasta ahora, agregar física a un LSTM bien entrenado no lo ha mejorado | Contrapeso: el híbrido se justifica por la extrapolación (PUR) y la interpretación, no por la NSE en PUB |

### 2.3 Transfer learning y datos de otras regiones

| Fuente | Hallazgo | Implicación |
|---|---|---|
| *DL generalization in glacierized basins* 2026 | La **representatividad hidroclimática** del entrenamiento pesa más que el tamaño de la muestra y que la arquitectura | Explica por qué esperamos peores resultados en el Pacífico y el Amazonas: casi no hay cuencas parecidas en el entrenamiento. Es el argumento para traer cuencas tropicales de afuera |
| Ouémé 2026 (`Enagnon97__TL-ORB`) | Preentrenar con las **50 cuencas de Caravan más parecidas** en clima tropical y luego ajustar es lo mejor. **Pesa más la similitud que el volumen** | 🔵 Receta concreta, con código para NeuralHydrology |
| Ma/Le et al. 2021 (WRR); Ougahi et al. 2026; Alzhanov et al. 2025 (`Almas-Alz__uba-transfer-learning`) | El transfer learning desde regiones ricas en datos mejora en Chile (49 cuencas), China y Asia Central. Con 150 cuencas de preentrenamiento se obtuvo más que con 300 (NSE 0,8187 frente a 0,7602 entrenando solo con datos locales) | 🔵 Requiere las mismas forzantes que Caravan (ERA5-Land) y un conjunto común de atributos (bitácora, sección 23) |
| CAMELS-PE 2026 (`hllauca__RCamelsPE`) | 136 cuencas de Perú, andinas tropicales y amazónicas, con atributos y forzantes en formato CAMELS | 🔵 **La fuente externa más parecida a la nuestra.** Sumarla al entrenamiento ataca la falta de representatividad del Amazonas y de los Andes. Hay que homologar las forzantes |
| Caravan 2023 (`kratzert__Caravan`) | Código para extraer ERA5-Land y HydroATLAS de cualquier polígono en Earth Engine | 🔵 Con él, CAMELS-COL quedaría en el formato de Caravan |
| Nearing et al. 2024 (Nature; `google-research-datasets__global_streamflow_model_paper`) | Un LSTM global da en cuencas sin estación pronósticos tan confiables como los de las regiones con estación | ⛔ No sirve como entrada: GRDC tiene caudal del IDEAM desde 2024 (bitácora, sección 23). Sirve como contexto y como argumento |
| Zhang et al. 2024 (The Innovation), ED-DLSTM | Codificador-decodificador LSTM entrenado por región (EE. UU. con 482 cuencas, Canadá con 740, Europa central), con atributos espaciales codificados; en la prueba, NSE mediano de 0,80 en EE. UU. Sin ajuste, aplicado a Chile desde el modelo de EE. UU., da NSE > 0 en el 76,9 % de las cuencas; desde Canadá en el 66,2 % y desde Europa central en el 53,1 % | Confirma que **la región de origen importa**: transferir sin ajuste entre climas distintos deja muchas cuencas peor que la media. Argumento para elegir donantes tropicales (Ouémé 2026, CAMELS-PE) y no "todo Caravan". 🔵 |
| Oruche et al. 2021 (Kenia) | Preentrenar un LSTM con CAMELS-US y ajustarlo con 26 estaciones de Kenia mejora frente a entrenar desde cero. **Agregar atributos estáticos no ayudó**, porque las fuentes de los atributos tenían escalas distintas en origen y destino | ⚠️ Lección directa para B1 y para cualquier transfer: los atributos deben salir de las **mismas fuentes** en todas las cuencas (por eso Caravan usa HydroATLAS para todas). Mezclar los 48 atributos de CAMELS-COL con los de otro dataset sin homologar puede anular su aporte |
| Oruche et al. 2023 | Atención y adaptación de dominio adversaria (un discriminador obliga a que las cuencas de origen y destino tengan representaciones parecidas). En CAMELS-CL supera al transfer learning por ajuste fino | ⛔ Es un pronóstico a 24 h que recibe caudal pasado. 🔵 La idea de alinear dominios sí aplicaría sin caudal, como trabajo futuro para el Pacífico y el Amazonas |
| Echternacht et al. 2025 (Sci. Rep.), CNN | Una CNN con transfer learning entre cuencas de Brasil y África: reduce hasta un 54 % el tiempo de entrenamiento con poca pérdida, y el contexto de los datos de preentrenamiento cambia el resultado | 🔵 No cambia la arquitectura que usamos (la literatura de PUB a gran escala favorece al LSTM). Sí suma evidencia a que **el contexto de los donantes pesa** |
| Sun 2026 (`dialuser__tsfm_study`); Liu 2025 (`thuml__Time-Series-Library`) | Los modelos fundacionales *zero-shot* todavía quedan detrás del LSTM entrenado cuando hay forzantes | 🔵 No priorizar |

### 2.4 Arquitecturas con atención (lo que pidió la profesora)

| Fuente | Hallazgo |
|---|---|
| Liu et al. 2025 (HESS) | En regresión diaria y en validación cruzada espacial, el LSTM gana a 11 Transformers. La atención gana en extremos y en pronóstico autorregresivo largo |
| Barbhuiya y Gupta 2025; Akinrele 2026 | Transformer, Informer, Reformer y Linformer quedan por debajo del LSTM. El LSTM también gana en la inferencia aguas arriba |
| Bakar et al. 2026; Koya 2023 | El TFT gana, pero en tareas temporales, y Bakar usa caudal rezagado |

**Acción 🟡:** un Transformer de NeuralHydrology (`model: transformer`) con el mismo protocolo, en el flujo espacial y con 1 semilla. Responde a la profesora con una comparación justa. La literatura anticipa que no gana, y ese resultado también es útil.

### 2.5 Información de cuencas vecinas o anidadas (⛔ como entrada; ✅ como referencia)

| Fuente | Hallazgo | Implicación |
|---|---|---|
| Wi et al. 2025 | Los donantes solo ayudan si su correlación de Spearman con la cuenca objetivo supera 0,9; si no, gana el LSTM que solo recibe clima | Nuestra referencia "vecino más cercano" mide esto en Colombia |
| Pool et al. 2021 | Hay que comparar contra una referencia de regionalización bien elegida (19 enfoques comparados) | ✅ Tres referencias: climatología de 5 donantes por atributos, caudal del vecino más cercano y LightGBM regional |
| Portugal 2026 (marginales) | La transferencia por distancia con IDW de varios donantes fue la mejor (NSE 0,81, mensual) | 🟡 Agregar IDW de 3 a 5 donantes como referencia más exigente |
| He 2026 (HEG-LSTM); Ramírez 2025; Akinrele 2026; GNN 2022 (`nnzhan__Graph-WaveNet`, `000Justin000__gnn-residual-correlation`) | El caudal de una estación anidada aguas abajo mejora mucho la estimación aguas arriba (+60 % de NNSE) | 🔵 Relevante porque 229 de nuestras 248 cuencas están anidadas. Sería otro producto: "cuenca parcialmente instrumentada" |
| Yu et al. 2024 (HESS), modelo SR | Aplicar el LSTM por subcuenca y tránsito hidráulico hasta la salida mejora las cuencas grandes no aforadas | 🔵 Útil para el visor: subcuencas pequeñas más tránsito |

### 2.6 Lo que dicen los trabajos colombianos

| Fuente | Hallazgo | Implicación |
|---|---|---|
| Jiménez et al. 2025 (CAMELS-COL) | 347 cuencas. CHIRPS subestima la lluvia en algunas regiones | Ya medido en la bitácora (secciones 10 y 13): es el techo de 32 cuencas |
| *Gap filling and trends* 2026 (HSJ) | Con RF y MLP mensuales, el desempeño depende de la escala: mejor en cuencas de más de 3000 km² y peor en las de 100 a 600 km² del Pacífico y los Andes | ✅ Reportar el NSE de la etapa 2 por rango de área; se espera el mismo patrón |
| *Hydropower basins* 2026 | En algunas cuencas colombianas el caudal rezagado es indispensable | Es el costo de nuestra regla, y hay que decirlo: en una cuenca sin estación ese dato no existe |
| Bernal (UNAL); Antioquia; Santander; IDEAM 2017 | La regionalización clásica en Colombia estima **caudales medios anuales, crecientes y curvas de duración**, no series diarias. El IDEAM definió 16 regiones hidrológicas en 9139 cuencas. En la Amazonía y la Orinoquía el flujo base subterráneo pesa mucho. En Urabá, el error de la regionalización fue del 44 % | **La aportación del proyecto es una serie diaria en cuencas sin estación, que la práctica colombiana no ofrece.** Comparar contra la curva de duración simulada, no solo contra el NSE |
| Pérez et al. 2026 (Caquetá) | Los parámetros de GR4J se regionalizan con atributos morfométricos (R² de 0,80 a 0,98) | Apoya la vía híbrida (2.2) en el Amazonas |
| Balance hídrico mensual y diario en cuencas colombianas 2010 (IAHR) | El modelo de Zhang et al. (2008), basado en Budyko, con solo 4 parámetros con sentido físico (caudales máximos, mínimos y amplitud), reproduce las series diarias y mensuales del río Aburrá. Los autores proponen regionalizar esos parámetros para cuencas sin información | 🟡 Es la referencia "clásica" diaria más cercana a la pregunta que hay en Colombia. Como trabajo futuro, sería un modelo conceptual regionalizado contra el cual comparar, o el núcleo físico de un híbrido (sección 2.2). En el artículo, cita de contexto |
| Orinoquía 2025; Tocaría 2026; Tuluá 2026; bajo Magdalena 2026 | Todos son de una cuenca, mensuales o con caudal observado | Confirman la brecha: no hay un estudio diario y regional de PUB para Colombia |

### 2.7 ML clásico (RF, SVM, XGBoost) como referencia contra el LSTM

| Fuente | Hallazgo | Implicación |
|---|---|---|
| Araza et al. 2020 (Luzón, Filipinas; `arnanaraza__streamflow`) | Random forest regionalizado en 21 cuencas tropicales de montaña. Agrupar primero (PCA y clústeres) y luego transferir el modelo es lo que mejor funciona | ⚠️ Evalúa con un 20 % de días al azar **de cada cuenca**, así que no es un verdadero PUB: hay fuga temporal. Sirve como contexto tropical, no como cifra comparable. Nuestro LightGBM sí se evalúa en cuencas retenidas |
| Singh et al. 2022 (HSJ), SVM y XGBoost | Regionalizan el caudal mensual de una cuenca sin estación de 688 km² en India con SVM y XGBoost, y son más precisos que los métodos lineales | ⛔ Reciben **el caudal aforado de la cuenca donante** como entrada, y es un caso de 2 cuencas. Equivale a nuestra referencia "vecino más cercano", no al modelo |
| Markhali et al. 2026 (Hydrol. Processes) | Un random forest que regionaliza **los parámetros** de un modelo hidrológico distribuido a partir de los atributos de la cuenca supera a la regresión lineal múltiple. Pasar de paso diario a 3 horas mejora la calibración | 🔵 Es la vía "atributos → parámetros", la misma de Pérez 2026 (Caquetá) y de los híbridos δ. Confirma que los atributos estáticos cargan la información para transferir (sección 2.1) |
| Le et al. 2022 (HESSD; `mhle510__PUR_experiments`) | SVM, RF y XGBoost entrenados en regiones ricas en datos (Norteamérica, Sudamérica, Europa occidental) para estimar el caudal **climatológico** de Sudáfrica y Asia Central. La incertidumbre de la extrapolación geográfica (PUR) es grande y depende de qué región se usa de origen | Coincide con Zhang 2024 y con el estudio de glaciares 2026: en PUR manda la representatividad del origen |
| Gauch et al. 2021 | Los árboles compiten con el LSTM cuando hay pocos datos; con más datos gana el LSTM | ✅ **Ya está en el plan: LightGBM regional** (familia XGBoost) con las mismas 83 variables, entrenado y evaluado con las mismas particiones (A3). Es la comparación justa que estos trabajos no hacen. Primer dato, partición 1: LightGBM 0,40 frente a LSTM 0,19 (1 semilla) |

---

## 3. Plan por fases

Presupuesto medido el 27/11 en la RTX 3050: **unos 11 minutos por época** con 229 cuencas, es decir, **unos 37 minutos por partición** con 3 épocas, más 3 de evaluación. Las 5 particiones salen en unas 3,3 horas por semilla.

### Fase A: flujo espacial base, la respuesta a la pregunta (en curso)

| # | Tarea | Estado / costo |
|---|---|---|
| A1 | Base reconstruida en esta máquina desde `04-datos/` (preprocesamiento, sección 1). Las listas de cuencas y particiones salen **idénticas** a las de git | ✅ Hecho el 27/11 |
| A2 | `src/flujo_espacial.py`: 5 particiones × semillas 42, 43 y 44, protocolo C1 (3 épocas fijas), más la corrida C1 de cada semilla como referencia de cuenca vista | ✅ Terminó el 27/11 a las 18:05 (18 corridas) |
| A3 | `src/baselines_espacial.py`: climatología de 5 donantes, vecino más cercano y LightGBM regional por partición | ✅ Corregido el 27/11: `gauge_lat` y `gauge_lon` vienen en metros de Mercator (EPSG:3395), no en grados |
| A4 | `src/resultados_espacial.py`: métricas en los mismos días, medianas, Wilcoxon, por región, por partición y la brecha entre cuenca vista y no vista (2016–2022, mismas cuencas) | ✅ 27/11 con 3 semillas; acepta `--variante era5l` |
| A5 | `src/figuras_espacial.py`: distribución acumulada del NSE, mapa y hidrogramas | ✅ 27/11, revisadas |
| A6 | ✅ 28/11. Llevar todo al notebook (secciones 7, 8 y 10) y agregar la sección 24 a la bitácora. Según `rules.md` (sección 6 de este plan): tabla única del checklist de fuga al inicio de la sección 7, una interpretación de 2–6 líneas después de cada tabla y figura de la sección 10, y el NSE presentado como R² | Celdas de las secciones 7, 8 y 10 listas el 27/11; la interpretación va después de A4 |
| A7 | Type hints y docstrings en las funciones de `src/*_espacial.py` (`rules.md`, regla 2) | ✅ 28/11, en los 5 scripts. Los resultados y las figuras salen idénticos |

**Cómo se lee el resultado (fijado antes de verlo):**
- Meta de la propuesta: NSE mediano ≥ 0,50 en las 95 cuencas no vistas.
- El LSTM **sirve para cuencas sin estación** si supera con claridad a la climatología de donantes (Wilcoxon) y al menos empata con el vecino más cercano. El vecino usa caudal observado de otra cuenca, así que es una referencia más fuerte que lo que el modelo puede usar.
- La brecha vista − no vista dice cuánto se pierde por no tener estación.
- Se reporta por región, por partición y por rango de área, y aparte las cuencas con balance inconsistente o intervenidas (bitácora, secciones 10 y 13).

### Fase B: mejoras con evidencia, en orden (28/11 a 30/11)

| # | Tarea | Por qué (sección 2) | Costo | Condición |
|---|---|---|---|---|
| B1 | **Etapa 2 con ERA5-Land (C2)**: 19 forzantes y 47 atributos | Fue lo único que movió el flujo temporal (+0,024). La lluvia de ERA5 corrige donde CHIRPS se queda corta (Atrato, Pacífico), que son justo las regiones retenidas | 18 corridas de unos 42 min | ✅ Terminó el 28/11 a las 10:14 (18 corridas). Cuenca no vista: 0,300 frente a 0,276 del base (54 de 95 cuencas mejoran, p = 0,08: tendencia, no significativa). Mejora en Caribe (0,29 frente a 0,26) y Orinoco (0,72 frente a 0,64); empeora en Magdalena-Cauca (−0,51 frente a −0,03, 6 cuencas). Cuenca vista: 0,480 frente a 0,425. Resultados en `runs/flujo_espacial/era5l/` y `docs/flujo-espacial-era5l-*` |
| B2 | IDW de 3 a 5 donantes como referencia | Pool 2021; Portugal 2026 | Minutos | Ninguna |
| B3 | Transformer de NeuralHydrology en la etapa 2, 1 semilla | Lo pidió la profesora; Liu 2025 | Unas 4 a 8 h (medir antes con una época) | Después de A2 |
| B4 | Agregados de forzantes como atributos: estacionalidad, bimodalidad de la lluvia, fracción de lluvia en cada temporada | *Better data or architecture* 2026 | 1 h, más 3,3 h de etapa 2 | Ninguna |
| B5 | Embeddings de AlphaEarth (notebook, sección 11) | *Better data or architecture* 2026 | Medio: cuenta de Earth Engine | Solo si B4 muestra que los atributos importan |
| B6 | Híbrido diferenciable (`hybrid_model`) | Feng 2023 y 2024: mejor en PUR | Medio-alto | Solo si sobra una noche; si no, queda como trabajo futuro con su justificación |
| B7 | **Crecientes por período de retorno** en las 95 cuencas retenidas: ajustar una distribución de máximos anuales al caudal observado de cada cuenca (períodos de retorno de 2, 5 y 10 años) y contar en cuántos eventos el LSTM, LightGBM y el vecino superan el mismo umbral (aciertos, falsas alarmas y F1). Es la métrica de `calculate_return_period_metrics.ipynb` de Nearing 2024 (sección 8) | El NSE y el FHV no dicen si el modelo **detecta** una creciente, que es lo que le importa al usuario del visor. Lee y Kim 2025: el LSTM en PUB subestima los picos | 1–2 h de CPU, sin GPU | ✅ 28/11 (`src/errores_espacial.py`; notebook 10.4). F1 del LSTM con ERA5-Land de 0,16 en crecientes de 2 años con umbral propio (recall del 15 %); 4 % con el umbral observado. Es el mejor modelo, pero **no sirve para alertas**. Con ±7 días, F1 = 0,26 |
| B8 | **Correlación de errores entre cuencas vecinas:** con los residuos diarios del LSTM en las cuencas retenidas, correlación entre pares de cuencas según la distancia (a 50, 100 y 200 km) y según estén en el mismo sistema de ríos. Es la idea de C-GNN (`000Justin000__gnn-residual-correlation`), usada como análisis y no como modelo | Si los errores de cuencas cercanas se parecen, el modelo falla por algo regional (por ejemplo, la lluvia de CHIRPS en el Pacífico) y no por la cuenca. Es evidencia para la discusión del artículo y para el trabajo futuro con cuencas anidadas (sección 2.5) | Menos de 1 h de CPU | ✅ 28/11. Mismo sistema y < 50 km: r mediana 0,38; sistemas distintos: 0,16; > 100 km: ≈ 0. Los errores son regionales y los p de Wilcoxon, optimistas |

Regla de decisión, igual que en la etapa 1: **cada mejora se elige con la validación interna de cada partición** (las cuencas de entrenamiento en 2011–2015) **o se declara de antemano**. Las cuencas retenidas no se usan para escoger.

### Fase C: despliegue para el usuario final (25 % de la nota)

La rúbrica del despliegue (sección 10) pesa por igual **funcionalidad** (recibe entradas, usa el modelo entrenado y entrega resultados sin errores) e **interfaz para el usuario final** (su lenguaje, resultados comprensibles y **auto contenida**: se usa sin explicación del equipo). La demo de la presentación vale además el 20 % de esa nota y pide **casos realistas**. Por eso se elige lo que funciona seguro, y no la versión más ambiciosa.

**Actualización 28/11: Sebastián ya tiene el visor "Caudal-COL" listo, pero no está en ningún repositorio.** Por el video del 17/11 (`03-codigo/visor/demo-visor-2026-11-17.mp4`), es una página web estática (`app_caudal_col/index.html`) con cuatro pestañas:
- **Mapa y cuenca:** filtros de región, cuenca, modelo y período; mapa de polígonos coloreado por región o por NSE; tarjetas de NSE, KGE, recall de crecientes y mejora frente a la línea base; hidrograma en m³/s o mm/día con lluvia y umbral de creciente; calidad del registro y aviso de confianza; descarga en CSV.
- **Comparar modelos:** tabla, CDF del NSE y NSE por región.
- **Calidad de datos:** banderas y cobertura por cuenca.
- **Acerca del modelo:** diagrama del LSTM y su configuración.

Ya resuelve casi toda la interfaz que pedía esta fase, y en lenguaje de usuario. **Se usa el visor y no se hace una app de Streamlit nueva.** Lo que le falta:
1. **Datos del LSTM:** en el video solo muestra la línea base (climatología) y dice "LSTM regional: en entrenamiento". Hay que darle las series y métricas de las etapas 1 y 2.
2. **"Ver como cuenca no aforada":** el interruptor existe con la etiqueta "pronto". Es justo la etapa 2: la simulación del modelo de la partición que no vio la cuenca (95 cuencas).
3. **Números viejos:** 187 cuencas en prueba (son 183) y meta 0,70 en todas las tarjetas. La meta de la etapa 2 es 0,50.
4. **"Usa el modelo entrenado" (50 % del despliegue):** si solo muestra series precalculadas, un evaluador puede decir que no usa el modelo. Hay dos opciones; la primera es la recomendada porque no necesita servidor y la página sigue siendo estática:
   - **ONNX en el navegador:** exportar el LSTM a ONNX (unos 1,3 MB por modelo) y correrlo con `onnxruntime-web` sobre las forzantes de la cuenca, con un control de **escenario de lluvia** (por ejemplo, −20 % para El Niño). El resultado se compara en el navegador con la serie precalculada, para comprobar que es el mismo modelo.
   - **API pequeña** (FastAPI en Hugging Face Spaces) que recibe la cuenca y el escenario y devuelve la serie.
5. **Publicación:** GitHub Pages, Netlify o Hugging Face Spaces, y probarla desde otro equipo.

La imagen `referencia-visor-rio-sinu.jpg` es de otro visor (Relieve 3D del río Sinú, Lab SinergIA) y solo sirvió de referencia visual.

**Usuario final:** un técnico de una corporación autónoma regional, una alcaldía o un operador de acueducto que necesita una estimación del caudal en un río sin estación. No es hidrólogo de modelos: piensa en m³/s, en meses secos y en crecientes.

**Qué hace la app (Streamlit, como el laboratorio 6):**
1. **Entrada:** el usuario escoge la cuenca en un mapa o en una lista por departamento y río (las 248 de CAMELS-COL), un rango de fechas y, opcionalmente, un **escenario de lluvia** (por ejemplo, −20 % para un año de El Niño, o +20 %).
2. **Modelo:** corre en CPU el LSTM de la etapa 2 que **no vio esa cuenca** (el de su partición, ensamble de 3 semillas). Así la app muestra de verdad una predicción "sin estación". Para las 153 cuencas que no están en ninguna partición se usa el LSTM entrenado con todas (C1), y la app lo dice.
3. **Salida, en el lenguaje del usuario:**
   - Hidrograma en **m³/s** (mm/día × área / 86,4), con el caudal observado superpuesto **solo como verificación** cuando existe.
   - Tres números: caudal medio del período, caudal de un mes seco típico y pico del período.
   - Una **confianza en palabras** según la región y el NSE medido en la etapa 2 (por ejemplo, "Orinoco: confiable, NSE típico 0,64"; "Magdalena-Cauca: poco confiable, úsese solo como orden de magnitud"), con una explicación de una línea de qué significa.
   - Descarga en CSV.
4. **Auto contenida:** un texto corto al inicio ("qué hace, qué no hace"), ayudas en cada control y un aviso fijo: el modelo no reemplaza una estación ni sirve para diseñar obras.

**Arquitectura:**
- Artefactos: los pesos `model_epoch003.pt` (unos 1,3 MB cada uno), `train_data_scaler.yml` y la configuración de cada partición; las forzantes y atributos de las 248 cuencas (unos 100 MB en `.nc`); `docs/flujo-espacial-cuencas.csv` para la confianza por cuenca.
- Inferencia con `neuralhydrology` o con el `CudaLSTM` cargado en PyTorch CPU; menos de 1 s por cuenca y año.
- Publicación en **Hugging Face Spaces** o **Streamlit Community Cloud** desde el repositorio, con `requirements.txt`. El artículo describe esta arquitectura en la metodología (la rúbrica lo pide).

**Fuera de alcance, como trabajo futuro:** un polígono dibujado por el usuario. Pide calcular los 48 atributos (suelos del IGAC, geología del SGC, número de curva) y forzantes en tiempo real (Open-Meteo, con la evapotranspiración de la sección 9.1). Se menciona en el artículo y en la presentación.

| # | Tarea | Cuándo |
|---|---|---|
| C0 | **Que Sebastián suba el visor** a una rama del repositorio del grupo (por ejemplo `visor` en `leodie1615/Proyecto-final-ADL`, carpeta `03-codigo/visor/` o `app/`), con una nota de qué archivos de datos lee y en qué formato | Hoy, 28/11 |
| C1 | Exportar al formato que lee el visor: series diarias y métricas por cuenca del LSTM de la etapa 1 y de la etapa 2 (cuenca no vista), con las tres referencias; y, si se elige ONNX, los modelos exportados más una prueba de que dan lo mismo que `series_espacial.parquet` | **ONNX ✅ 28/11** (`src/exportar_modelo.py`): los 15 LSTM de ERA5-Land (5 particiones × 3 semillas) en `despliegue/modelos/`, 1,3 MB cada uno, con un JSON de entradas y cuencas no vistas. La envoltura recibe forzantes y atributos crudos y normaliza sola. Verificada contra la evaluación de NeuralHydrology: diferencia media de unos 1e-4 mm/día (máx. 0,035) por calcular en CPU; ONNX = PyTorch con diferencias por debajo de 3e-6. Ojo: NeuralHydrology ordena los atributos alfabéticamente. **Falta:** las series en el formato del visor, cuando Sebastián lo suba |
| C2 | Conectar los datos al visor: el selector de modelo, el interruptor de cuenca no aforada, la meta de 0,50 en la etapa 2 y el escenario de lluvia. Lo hace Sebastián o nosotros sobre su rama | 29/11 – 30/11 |
| C3 | Publicarlo y probarlo de punta a punta en un equipo sin el repositorio | 30/11 |
| C4 | Tres casos realistas para la demo: una cuenca del Orinoco (funciona bien), una del Caribe (media) y un escenario de El Niño | 1/12 |

### Fase D: artículo y presentación (1/12 a 4/12)

- **Estado del arte:** se arma con las dos revisiones. *AI for streamflow in ungauged watersheds* (Water, 2025) ordena el campo en tres frentes: regionalización por similitud y transfer learning, datos sustitutos (NDVI, humedad del suelo, DEM) y evaluación; concluye que la IA supera a los modelos físicos y conceptuales en cuencas sin estación. El *survey* de ML para sitios no monitoreados (Environmental Data Science, 2024) clasifica los métodos en modelos que conocen la entidad (LSTM con atributos estáticos, que es el nuestro), transfer learning y ML guiado por conocimiento (híbridos). Cada fila de la sección 2 cae en una de esas tres familias.
- **Pregunta del artículo:** la del título de este plan. **Objetivo 3 (etapa 2):** pasa de "resultado esperado" a resultado medido.
- **Números que hay que actualizar en el `.tex`:** de 346 cuencas a 248; de 44 sistemas a 35; de 5 particiones de 26 a 5 de 19; de 216 fijas a 153; de 252 con prueba a 183.
- **Hiperparámetros (`rules.md`, regla 6):** tabla top-N de lo que sí se ajustó (`hidden_size` y épocas), con media ± desviación entre semillas, el score de prueba del elegido y 2–4 líneas de por qué ganó. Explicar por qué no hubo `RandomSearch` (costo; laboratorio 2) ni parada temprana en la etapa 2 (fuga).
- **Resultados:** tabla de la etapa 1 (ya existe), tabla de la etapa 2 con sus tres referencias, figuras de A5 y desempeño por región.
- **Discusión:** representatividad (glaciares 2026), datos de entrada (CHIRPS, ERA5-Land), intervención humana, la regla de no usar caudal (*hydropower* 2026) y la comparación con la regionalización clásica colombiana (2.6).
- **Trabajo futuro, con su evidencia:** CAMELS-PE y Caravan por similitud climática, híbridos δ, AlphaEarth, información de cuencas anidadas y curvas de duración regionales.
- **Presentación de 12 minutos más 5 de preguntas** (estricto). La rúbrica pesa problema 20 %, solución 25 %, resultados 25 %, demo 20 % y tiempo 10 %. Reparto sugerido: problema 2 min (El Niño 2026–27, estaciones escasas), solución 3 min (datos, LSTM regional sin caudal, validación espacial), resultados 3 min (CDF y mapa de la etapa 2, brecha vista − no vista, limitaciones), demo 3 min (los casos de C4), cierre 1 min. Sin leer las diapositivas. El `.pptx` va al padlet antes de las 5:30 pm del 2/12.
- **Artículo:** plantilla IEEE (`02-entregables/articulo-latex/bare_jrnl.tex`, hoy 4 páginas con la estructura de la propuesta). Hay que reestructurarlo con los apartados de la rúbrica (sección 10). Meta: **10 páginas con referencias**; la rúbrica tolera hasta 12, y la declaración de uso de IA generativa y los anexos no cuentan. Va también al padlet.

---

## 4. Calendario

| Fecha | Qué |
|---|---|
| Vie 27/11 | A1–A5. A2 termina de noche. Pedir a Diego `data/processed_era5land` |
| Sáb 28/11 | B1 termina (~10:00); resultados y figuras de ERA5-Land. A6 (notebook, bitácora 24, README). B7 y B8 en CPU. Lanzar B3 de noche (ojo: el PC se apaga a las 20:00 por `apagado-nocturno`) |
| Dom 29/11 | C1 y C2 (app). B2 y B4 en CPU |
| Lun 30/11 | **Congelar resultados.** C3: app publicada. Bitácora y notebook al día. Empezar el artículo |
| Mar 1/12 | Diapositivas y C4 (casos de la demo). Ensayo cronometrado de 12 minutos |
| Mié 2/12 | Presentación, 5:30 pm (`.pptx` en el padlet antes) |
| Jue 3/12 – Vie 4/12 | Artículo final (meta 10 páginas con referencias; tope 12), por el enlace y en el padlet |

## 5. Riesgos

| Riesgo | Mitigación |
|---|---|
| La etapa 2 queda por debajo de 0,50 | Es un resultado válido. Se reporta con las referencias, por región y con la brecha vista − no vista; la literatura (secciones 2.2 y 2.3) explica por qué y qué haría falta |
| Los datos de ERA5-Land no llegan a esta máquina | La etapa 2 base (C1) ya responde la pregunta. B1 queda como mejora documentada |
| Corte de luz o suspensión | `flujo_espacial.py` salta las corridas evaluadas; el entrenamiento corre con `systemd-inhibit`. El temporizador `apagado-nocturno` apaga el PC a las 20:00 (pasó el 27/11): las corridas largas se lanzan después del apagado o se pausa el temporizador |
| La app falla en la demo | Se prueba publicada desde otro equipo (C3) y se graba un video de respaldo de los casos de C4 |
| Fuga de información | Las particiones usan sistemas de ríos completos (verificado en el notebook, sección 4.4). La estandarización y la validación usan solo las cuencas de entrenamiento de cada partición. Las cuencas retenidas no deciden nada |

---

## 6. Cumplimiento de `rules.md` (revisado el 27/11)

Se revisaron el código de `src/`, los dos notebooks y este plan contra cada regla. ✅ cumple, ⚠️ cumple a medias (tarea abajo), ❌ no cumple.

| Regla | Estado | Evidencia / tarea |
|---|---|---|
| **1. Estructura:** cada sección con "qué se hace", código y "resultado + interpretación"; orden exploración → preprocesamiento → modelado → evaluación → análisis | ⚠️ | El orden se cumple entre notebooks: exploración en `analisis-camels-col/analisis_detallado_camels_col.ipynb`, preprocesamiento y luego modelado (secciones 1–8 de modelado, evaluación en 9 y 10, análisis auxiliar en 11). **Falta:** la sección 10 no tiene todavía su celda de interpretación, porque no hay resultados. **Tarea A6:** una celda de 2–6 líneas después de cada tabla y figura de la sección 10, que diga qué significa el número y por qué importa (meta 0,50, brecha vista − no vista, regiones) |
| **2. Código tipado,** con type hints y docstring de una línea; sin `fit()` repetido ni código muerto | ✅ desde el 28/11 (antes ❌) | 29 funciones de los 4 scripts nuevos no tienen type hints completos y 17 no tienen docstring. **Tarea A7:** completarlos cuando termine el entrenamiento (no se edita `flujo_espacial.py` con corridas en curso). Las 18 corridas no son `fit()` repetidos: son 5 particiones × 3 semillas, más la referencia, y cada una entrena un modelo distinto. Cada celda del notebook lanza el script una sola vez y está protegida con su bandera `EJECUTAR_*` |
| **3.1 Diccionario de datos** (target, features, faltantes, rangos) | ✅ | Preprocesamiento, sección 1.1 (4 forzantes, 48 atributos y el target `streamflow` en mm/día), 1.4 (caudal faltante), 1.5 (huecos de forzantes) y 1.7 (atributos casi vacíos). La distribución del target está en el notebook de exploración |
| **3.2 Checklist de fuga** con decisión explícita | ✅ desde el 28/11: tabla al inicio de la sección 7 del notebook (antes ⚠️) | Las decisiones existen, pero repartidas: firmas hidrológicas eliminadas (derivan del caudal), coordenadas eliminadas (identifican la cuenca), índices climáticos recalculados solo con 1981–2010 (los publicados usan el período de prueba), particiones por sistema de ríos (evitan fuga por anidamiento) y regla de no usar caudal como entrada. **Tarea A6:** juntarlas en una tabla única "feature → ¿fuga? → se elimina o se mantiene → por qué" al inicio de la sección 7 del notebook de modelado, con la fila nueva de la etapa 2: las referencias de donantes y del vecino **sí** usan caudal de otras cuencas, y por eso solo son referencia, nunca entrada |
| **4. Preprocesamiento dentro del pipeline,** nunca ajustado con todo el dataset | ✅ | Preprocesamiento, sección 1.10: no se estandariza nada fuera del modelo. NeuralHydrology calcula media y desviación solo con las cuencas y el período de entrenamiento de cada corrida (`train_data_scaler.yml`), así que en cada partición se recalculan sin las 19 retenidas. En `baselines_espacial.py`, la similitud de atributos se estandariza solo con las cuencas de entrenamiento de la partición (línea 138) |
| **5. Evaluación de regresión:** RMSE, MAE y R², cada una con interpretación | ✅ con una aclaración | Se reportan NSE, KGE, RMSE, MAE, r, α, β, FHV, FLV y FMS. **El NSE es exactamente el R² de `sklearn.metrics.r2_score`** (1 − SSE/SST), calculado por cuenca. Decirlo así en el notebook y en el artículo: "NSE (equivalente al R²)". Se usa la mediana entre cuencas, y no el R² agregado, porque una cuenca grande dominaría la suma de cuadrados. **Tarea A6:** interpretar cada métrica en unidades del problema (por ejemplo, "un MAE de x mm/día equivale a y m³/s en una cuenca de 1000 km²") |
| **6. Búsqueda de hiperparámetros:** `best_score_`, score en test, tabla top-N con media y desviación, y por qué ganaron | ⚠️ | No se hizo una búsqueda amplia, y está justificado (bitácora, sección 2): cada corrida cuesta unos 40 minutos en una GPU de 6 GB, y la configuración viene de la literatura (Kratzert 2019, Wi 2025). Lo que **sí** se ajustó, `hidden_size` y las épocas (bitácora, secciones 4 y 7, y la sección 6.6 del notebook), debe reportarse con el formato de la regla. **Tarea D:** tabla top-N con NSE de validación medio ± desviación entre semillas, NSE de prueba del elegido y 2–4 líneas de por qué ganó (capacidad frente a sobreajuste: con 3 épocas y dropout de 0,4, más unidades no sobreajustan). En la etapa 2 no se ajusta nada con las cuencas retenidas, y esa es la justificación |
| **7. Visualizaciones** con título, ejes y `tight_layout()`; valores impresos por consola | ✅ | `figuras_espacial.py` llama `tight_layout()` en las 3 figuras. `resultados_espacial.py` imprime y guarda las tablas en CSV. Revisar al generarlas que cada eje tenga sus unidades (NSE adimensional, caudal en mm/día) |
| **La regla más importante: el porqué de cada decisión** | ✅ | Cada fila de la sección 2 trae su "implicación" y cada tarea de la fase B su "por qué". Las decisiones de la etapa 2 se fijaron antes de ver resultados (sección 3, fase A) |

## 7. Lo que se aprovecha de los laboratorios del curso

Solo se toman los laboratorios que aportan algo a esta pregunta. Los de imágenes (L1, L3, L5, L5.1) y de texto generativo (L8, L10) no cambian el plan.

| Laboratorio | Qué se vio | Dónde se usa en el plan |
|---|---|---|
| **L6: LSTM + despliegue** | LSTM con `EarlyStopping` y `Dropout`, y despliegue en **Streamlit** con `app.py`, `requirements.txt`, artefactos del modelo y repositorio en GitHub | **Fase C.** Es el patrón de despliegue que el curso enseñó, así que es la opción más segura para el 25 % de la nota: una app de Streamlit que recibe la cuenca (o un polígono de las 23 estaciones de Cundinamarca), baja las forzantes, corre el LSTM y muestra el hidrograma con su banda de confianza por región. Los artefactos serían el `model_epoch003.pt`, el `train_data_scaler.yml` y la configuración. Si el visor del equipo (`03-codigo/visor`) ya resuelve la interfaz, Streamlit queda como respaldo |
| **L2: optimización e hiperparámetros** | `keras_tuner.RandomSearch`, `EarlyStopping`, dropout y tasa de aprendizaje | **Regla 6 y fase D.** Justifica en el artículo por qué no se hizo una `RandomSearch`: con unos 40 minutos por corrida, 20 combinaciones × 5 particiones serían unas 67 horas. Justifica también por qué la etapa 2 usa 3 épocas fijas y no `EarlyStopping`: la parada temprana necesita un conjunto de validación, y si fueran las cuencas retenidas habría fuga |
| **L4: modelo preentrenado** | Transfer learning con capas congeladas (`trainable = False`), luego ajuste fino con `ReduceLROnPlateau` | **Sección 2.3, trabajo futuro.** Es la misma receta de Ouémé 2026 y Alzhanov 2025: preentrenar el LSTM con cuencas tropicales de Caravan o CAMELS-PE, congelar la capa LSTM y ajustar la capa de salida con CAMELS-COL. En NeuralHydrology se hace con `finetune_modules` |
| **L7: BERT + BertViz** | Mapas de atención para interpretar qué tokens mira el modelo | **B3 (Transformer).** Si se entrena el Transformer, se grafican sus pesos de atención sobre la ventana de 365 días en 2–3 cuencas retenidas: muestra qué días antes de un pico mira el modelo. Le da a la comparación pedida por la profesora una parte interpretativa, además de la métrica |
| **L9: ViT frente a CNN** | Comparar arquitecturas en igualdad de condiciones, como extractores de características | **B3.** Mismo principio: Transformer y LSTM con las mismas entradas, particiones, épocas y semilla. Si no, la comparación no vale |
| **L11: CLIP, zero-shot y sonda lineal** | Usar embeddings de un modelo fundacional sin reentrenarlo (zero-shot) y una sonda lineal encima | **B5 y la sección 11 del notebook.** Los embeddings de AlphaEarth reducidos con PCA son exactamente eso: representaciones preentrenadas usadas como atributos. La sonda lineal da una prueba barata antes de reentrenar el LSTM: si una regresión lineal de los embeddings no predice firmas como el caudal medio en las cuencas de entrenamiento, no vale la pena B5. El zero-shot de modelos fundacionales (Sun 2026) ya está descartado en 2.3 |

## 8. Qué se hace con cada repositorio de `05-referencias/repos/`

Para cada repositorio: si **se corre su código** en este proyecto, si se toma **solo una idea o un diseño**, o si **no se usa**, y por qué. Leyenda igual que en la sección 2.

| Repositorio | Paper | ¿Se corre su código? | Uso concreto en el plan |
|---|---|---|---|
| `frameworks-y-datasets/neuralhydrology__neuralhydrology` (y `neuralhydrology/`, que es una copia idéntica: mismo commit `ab4df04`, versión 1.13.0) | Librería base | ✅ **Sí, es el motor de todo el proyecto.** Se instala la versión 1.13.0 en `.venv` | Fase A: `cudalstm` con `nh_run train/evaluate` desde `src/flujo_espacial.py`. B3: `model: transformer`. B6: `hybrid_model`. 🟡 `ealstm`. 🔵 Transfer learning con `finetune_modules` (laboratorio 4). El clon sirve para leer el código fuente al configurar (por ejemplo, cómo calcula el escalador con solo las cuencas de entrenamiento) |
| `lstm-regional/kratzert__lstm_for_pub` | Kratzert 2019 (WRR) | No: es código de 2019 anterior a NeuralHydrology, para CAMELS-US | ✅ **Se toma su diseño de validación**: *k-fold* espacial, en el que cada cuenca se simula con un modelo que no la vio, y ensamble de reinicios. **Se cambia una cosa:** ellos parten las cuencas al azar en 12 grupos, y nosotros por sistema de ríos en 5, porque 229 de las 248 cuencas están anidadas y una partición al azar dejaría cuencas conectadas a ambos lados (fuga). Usamos 3 semillas y no 10, por costo |
| `lstm-regional/kratzert__ealstm_regional_modeling` | Kratzert 2019 (HESS), EA-LSTM | No: se usa la `ealstm` de NeuralHydrology, que la reemplaza | 🟡 EA-LSTM para ver en qué cuencas de entrenamiento "se apoya" cada cuenca no vista, mediante la similitud de sus compuertas de entrada |
| `lstm-regional/gauchm__ealstm_regional_modeling` | Gauch 2021, *Proper care and feeding of CAMELS* | No | ✅ Justifica que la referencia de árboles (LightGBM) es necesaria: su `main_xgboost.py` muestra que XGBoost compite con la EA-LSTM cuando hay pocos datos. 🔵 Su experimento de **cantidad de datos de entrenamiento** (`main_gridEvaluation.py`) se podría repetir como curva de aprendizaje: NSE de la etapa 2 según cuántas cuencas de entrenamiento se usan |
| `lstm-regional/francisconog__RegBaciasCE` | Nogueira Filho 2022 (Ceará) | No | ✅ Refuerza dos decisiones: comparar contra modelos simples y entrenar con todas las cuencas |
| `articulo-citadas/Almas-Alz__uba-transfer-learning` | Alzhanov 2025 | No por ahora | 🔵 Sus `configs/` (`lstm-l`, `lstm-nc`, `lstm-ncl`) son archivos `.yml` de NeuralHydrology para preentrenar con Caravan y ajustar con datos locales: **sirven de plantilla** si se hace el transfer learning (sección 2.3) |
| `transfer-y-global/Enagnon97__TL-ORB` | Ouémé 2026 | No por ahora | 🔵 Receta para elegir donantes: un índice de similitud climática (lluvia media anual y estacionalidad) para escoger las 50 cuencas tropicales de Caravan más parecidas, preentrenar y luego ajustar. Sus `configs/` también son de NeuralHydrology |
| `frameworks-y-datasets/kratzert__Caravan` | Caravan 2023 | No por ahora | 🔵 Código de Earth Engine para extraer ERA5-Land y HydroATLAS de los polígonos de CAMELS-COL y dejarlos en formato Caravan. ⚠️ **También es un plan de respaldo para B1** si no llega `processed_era5land` de Diego, pero no es equivalente: daría otras variables y otra agregación, y obligaría a rehacer la configuración `era5l`. Requiere una cuenta de Earth Engine |
| `frameworks-y-datasets/hllauca__RCamelsPE` | CAMELS-PE 2026 | No por ahora | 🔵 Acceso a las 136 cuencas de Perú para sumarlas al entrenamiento y dar representatividad al Amazonas y los Andes. Hay que homologar forzantes y atributos (Oruche 2021) |
| `transfer-y-global/google-research-datasets__global_streamflow_model_paper` | Nearing 2024 | No se corren sus modelos | ⛔ Sus simulaciones no sirven como referencia justa: el modelo global se entrenó con estaciones de GRDC, que incluye caudal del IDEAM, así que nuestras cuencas "no vistas" podrían haber estado en su entrenamiento. 🟡 Sí se toma una idea de evaluación: `calculate_return_period_metrics.ipynb` mide aciertos en crecientes por período de retorno, que complementaría el FHV para decir si el LSTM detecta las crecientes en cuencas sin estación. Es barato porque usa las series que ya produce la fase A. Su `create_ungauged_experiments_gauge_groups.ipynb` confirma el diseño de grupos de estaciones retenidas |
| `articulo-citadas/DARE-ML__DeepGR4J-Extremes` | Kapoor 2026 | No | 🔵 Banda de incertidumbre por cuantiles para el despliegue (fase C), como trabajo futuro |
| `regionalizacion-y-ml-clasico/mhle510__PUR_experiments` | Le 2022 | No: son scripts de R con SVM, RF y XGBoost | ✅ La comparación de ML clásico ya se cubre con LightGBM en Python, que es de la misma familia que XGBoost, con las mismas particiones. Se toma su diseño de análisis: medir cómo cambia el error según la región de origen. Aquí equivale a **reportar el NSE por partición** (A4), porque cada partición deja fuera una mezcla distinta de regiones |
| `regionalizacion-y-ml-clasico/arnanaraza__streamflow` | Araza 2020 (Luzón) | **No se usa** | Scripts de R con rutas fijas (`D:/THESIS_PP`) y una evaluación con días al azar de cada cuenca, que tiene fuga temporal. Queda solo como contexto tropical en la sección 2.7 |
| `regionalizacion-y-ml-clasico/DassHydro__smash` | Huynh 2023 | **No, en este plazo** | smash es un modelo hidrológico distribuido (Python + Fortran) que necesita forzantes en malla y una red de drenaje por celdas, no series por cuenca como las nuestras. 🔵 Si B6 (`hybrid_model` de NeuralHydrology) funciona, smash sería el paso siguiente: un híbrido distribuido que regionaliza parámetros con una red neuronal. CAMELS-COL trae la red de drenaje (`04-datos/12_CAMELS_COL_Drainage_network.zip`) |
| `regionalizacion-y-ml-clasico/nghi-truyen__Regionalization-Learning` | Huynh 2023 | **No, en este plazo** | Scripts sobre smash (`run_regionalization.py`) con el método HDA-PR. 🔵 Lo mismo que smash: es la receta concreta si se hiciera la regionalización híbrida distribuida |
| `fisico-hibrido-y-grafos/nnzhan__Graph-WaveNet` | GNN 2022 (HESS) | **No** | ⛔ El código es de tráfico (METR-LA) y, aplicado a ríos, recibe el caudal observado de las estaciones vecinas del grafo como entrada. Eso choca con la regla del proyecto. 🔵 Sería otro producto, la "cuenca parcialmente instrumentada" (sección 2.5) |
| `fisico-hibrido-y-grafos/000Justin000__gnn-residual-correlation` | GNN 2022 (HESS), C-GNN (KDD 2020) | **No** | ⛔ Corrige la predicción de un nodo con los residuos observados de sus vecinos, así que también necesita caudal observado en estaciones vecinas (en Julia). 🔵 La idea sí sirve como análisis sin tocar el modelo: medir si **los errores del LSTM en cuencas retenidas están correlacionados con los de sus vecinas**. Si lo están, es evidencia a favor del trabajo futuro con información de cuencas anidadas |
| `transformer-y-modelos-fundacionales/thuml__Time-Series-Library` | Liu 2025 | **No** | B3 usa el `transformer` de NeuralHydrology, no esta librería. La razón: NeuralHydrology ya recibe los atributos estáticos de cada cuenca, usa los mismos datos, la misma pérdida (NSE\*) y el mismo protocolo que el LSTM, así que la comparación es justa (laboratorio 9). Los modelos de TSLib (Informer, PatchTST, etc.) están hechos para pronóstico de series sin atributos por cuenca y habría que adaptar la carga de datos. 🔵 Solo si se quisiera comparar más variantes de Transformer |
| `transformer-y-modelos-fundacionales/dialuser__tsfm_study` | Sun 2026 | **No** | ⛔ Los modelos fundacionales de series de tiempo pronostican el caudal a partir de su propia historia (univariado): en una cuenca sin estación esa historia no existe. Su resultado (no superan al LSTM entrenado con forzantes) solo se cita |

**Resumen:** en este plazo solo se corre código de NeuralHydrology. Se toman cuatro diseños (*k-fold* espacial de `lstm_for_pub`; referencia de árboles de Gauch 2021; NSE por partición de `PUR_experiments`; crecientes por período de retorno de Nearing 2024) y una idea de análisis (correlación de residuos entre vecinas, de C-GNN). Cinco quedan como plantillas para el trabajo futuro: los de transfer learning (`uba-transfer-learning`, `TL-ORB`, Caravan, CAMELS-PE) y la banda por cuantiles de DeepGR4J. Los demás no se usan, con la razón escrita.

## 9. Fórmulas que usa el plan

### 9.1 Evapotranspiración potencial (`poten_evapo`): Hargreaves-Samani con la radiación de FAO-56

Es la única forzante que se calcula, en lugar de leerse de una fuente: entra al LSTM todos los días y hay que poder recalcularla igual en el despliegue (fase C, punto 3). La implementación es `radiacion_extraterrestre_mm` del notebook de análisis (`analisis-camels-col/analisis_detallado_camels_col.ipynb`, celda 36).

**Radiación extraterrestre diaria** (FAO-56, Allen et al. 1998, ecuaciones 21 a 25), con *J* el día del año (1 a 365) y φ la latitud del centroide de la cuenca:

$$\varphi\,[\text{rad}] = \frac{\pi}{180}\,\varphi\,[°] \qquad \text{(ec. 22; positiva en el hemisferio norte)}$$

$$d_r = 1 + 0{,}033\cos\!\left(\frac{2\pi J}{365}\right) \qquad \text{(ec. 23, distancia relativa inversa Tierra-Sol)}$$

$$\delta = 0{,}409\,\sin\!\left(\frac{2\pi J}{365} - 1{,}39\right) \qquad \text{(ec. 24, declinación solar, rad)}$$

$$\omega_s = \arccos\!\left(-\tan\varphi\,\tan\delta\right) \qquad \text{(ec. 25, ángulo a la puesta del sol, rad)}$$

$$R_a = \frac{24\cdot 60}{\pi}\,G_{sc}\,d_r\left[\omega_s\sin\varphi\sin\delta + \cos\varphi\cos\delta\sin\omega_s\right] \qquad \text{(ec. 21)}$$

con $G_{sc} = 0{,}0820$ MJ m⁻² min⁻¹ (constante solar) y $R_a$ en MJ m⁻² día⁻¹. En la ecuación 21, el último seno es de $\omega_s$.

**Conversión a evaporación equivalente** (FAO-56, ecuación 20):

$$R_a' \,[\text{mm día}^{-1}] = 0{,}408 \cdot R_a \,[\text{MJ m}^{-2}\,\text{día}^{-1}]$$

**Hargreaves-Samani** (Hargreaves y Samani, 1985):

$$ET_o = C_o \, R_a' \,(T_{max} - T_{min})^{0{,}5}\,(T + 17{,}8), \qquad T = \frac{T_{max} + T_{min}}{2}, \qquad C_o = 0{,}0023$$

con temperaturas en °C y $ET_o$ en mm/día. En el código, $T_{max} - T_{min}$ se recorta en 0 para que la raíz exista si un día los datos vienen invertidos.

**Por qué importa el 0,408 (CORRECCIONES, sección 1):** CAMELS-COL publicó la ETP usando $R_a$ en MJ, sin pasarla a mm. Eso la infla por 1/0,408 = 2,45 (media de 8,9 mm/día frente a 3,3 corregida), con una correlación diaria de 0,997 con la fórmula correcta: es la misma fórmula con otra escala. En el LSTM casi no afecta, porque la estandarización z-score absorbe un factor constante. Pero sí afecta a los atributos `pet_mean` y `aridity`: con la ETP inflada, la aridez mediana sería 1,62 (cuencas secas) y no 0,59 (húmedas). Por eso el preprocesamiento los recalcula (sección 1.8).

**Por qué $C_o$ fijo en 0,0023 y no calibrado:** algunos trabajos calibran $C_o$ en cada sitio, minimizando el RMSE frente a FAO Penman-Monteith. Eso exige radiación, humedad y viento medidos en cada cuenca, que no existen en una cuenca sin estación. Con el valor fijo, la misma fórmula sirve en las 248 cuencas y en el despliegue. 🔵 Como trabajo futuro, ERA5-Land trae radiación, humedad y viento, así que se podría calcular Penman-Monteith en todas partes.

### 9.2 Métricas de evaluación (por cuenca, en los mismos días válidos; se reporta la mediana entre cuencas)

Con $Q_o$ el caudal observado, $Q_s$ el simulado, μ las medias y σ las desviaciones en los días evaluados:

$$\text{NSE} = 1 - \frac{\sum_t (Q_s^t - Q_o^t)^2}{\sum_t (Q_o^t - \mu_o)^2} \qquad \text{(igual al } R^2 \text{ de } \texttt{sklearn.metrics.r2\_score}\text{)}$$

NSE = 1 es perfecto; NSE = 0 equivale a predecir siempre la media observada de la cuenca. Por eso se puede leer como "cuánto mejor que la media"; un valor negativo es peor que la media.

$$\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}, \qquad \alpha = \frac{\sigma_s}{\sigma_o}, \qquad \beta = \frac{\mu_s}{\mu_o}$$

con *r* la correlación de Pearson. Separa tres errores: el tiempo (*r*), la variabilidad (α < 1: hidrograma muy plano) y el volumen (β ≠ 1: sesgo).

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_t (Q_s^t - Q_o^t)^2}, \qquad \text{MAE} = \frac{1}{n}\sum_t |Q_s^t - Q_o^t| \qquad \text{(mm/día)}$$

Para pasar a caudal: $Q\,[\text{m}^3/\text{s}] = Q\,[\text{mm/día}] \cdot A\,[\text{km}^2] / 86{,}4$.

FHV y FLV (Yilmaz et al., 2008) son el sesgo porcentual del volumen en el 2 % de caudales más altos y en el 30 % más bajo de la curva de duración. FHV < 0 significa que los picos se subestiman.

### 9.3 Pérdida de entrenamiento: NSE\* (Kratzert et al., 2019)

$$\text{NSE}^* = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_b}\sum_{t=1}^{N_b}\frac{(\hat{y}_{b,t} - y_{b,t})^2}{(\sigma_b + \epsilon)^2}, \qquad \epsilon = 0{,}1$$

con $\sigma_b$ la desviación del caudal de la cuenca *b* **en el período de entrenamiento**. Divide el error de cada cuenca por su propia variabilidad, para que las cuencas muy caudalosas no dominen la pérdida. Es la misma ponderación que se le da a LightGBM: peso $1/(\sigma_b + 0{,}1)^2$ por fila (`baselines_espacial.py`).

### 9.4 Coordenadas de las estaciones: de World Mercator (EPSG:3395) a grados

`gauge_lat` y `gauge_lon` de CAMELS-COL están en metros de Mercator elipsoidal (WGS84: $a = 6\,378\,137$ m, $e = 0{,}08181919$). La conversión (`mercator_a_grados` en `baselines_espacial.py`) es:

$$\lambda = \frac{x}{a}, \qquad t = e^{-y/a}, \qquad \varphi_{k+1} = \frac{\pi}{2} - 2\arctan\!\left[t\left(\frac{1 - e\sin\varphi_k}{1 + e\sin\varphi_k}\right)^{e/2}\right]$$

empezando en $\varphi_0 = \pi/2 - 2\arctan t$ y con 10 iteraciones (converge en 3 o 4). Con los grados, la distancia al vecino más cercano se mide con la fórmula de haversine, con un radio de 6371 km.

---

## 10. Cumplimiento de la rúbrica de la entrega final (revisado el 28/11)

Fuente: `01-enunciado/entrega-final/entrega-final.md`. La entrega final vale el 30 % del curso: **artículo 60 %, despliegue 25 %, presentación 15 %**. ✅ cubierto, ⚠️ cubierto a medias (tarea), ❌ falta.

### 10.1 Artículo (60 %)

| Criterio (peso) | Estado | Qué hay y qué falta |
|---|---|---|
| **Planteamiento del problema (10 %):** definido, delimitado, importancia con evidencia, objetivos verificables | ⚠️ | El `.tex` tiene contexto y objetivos de la propuesta. **Falta:** delimitar a "caudal medio diario en cuencas sin estación, 248 cuencas de CAMELS-COL, 1981–2022"; reescribir los objetivos para que se pueda decir si se cumplieron (por ejemplo, "NSE mediano ≥ 0,50 en cuencas no vistas": no se cumplió, 0,276), y argumentar la importancia con cifras (cuántas estaciones activas hay frente a cuántos ríos; El Niño 2026–27) |
| **Estado del arte (10 %):** crítico, actualizado, compara y sitúa el proyecto | ⚠️ | El material está en la sección 2 de este plan, con implicación por familia. **Falta:** escribirlo como comparación, no como lista: una tabla de trabajos (región, n.º de cuencas, validación, NSE en cuenca no vista) con nuestra fila al final, y la brecha: no hay un LSTM regional evaluado en cuencas no vistas de Colombia con validación por sistema de ríos (sección 2.6) |
| **Metodología (25 %):** datos (fuente, características, calidad), preprocesamiento, técnicas, arquitectura, entrenamiento, **ajuste de hiperparámetros**, **esquema de validación**, herramientas y entorno, decisiones justificadas, **arquitectura de despliegue** | ⚠️ | Todo existe en la bitácora y los notebooks. **Falta:** la tabla top-N de hiperparámetros (sección 6, regla 6); un diagrama del flujo (datos → preprocesamiento → LSTM → validación espacial por sistema de ríos → app); la tabla del checklist de fuga (A6); el entorno (Python 3.11, PyTorch, NeuralHydrology 1.13, RTX 3050 de 6 GB, ~11 min por época), y la arquitectura de la app (fase C) |
| **Resultados y análisis (20 %):** métricas adecuadas, análisis cuantitativo **y cualitativo** (errores, casos representativos), visualizaciones legibles **comentadas en el texto** | ⚠️ | Cuantitativo listo: etapa 1, etapa 2 con 3 referencias, por región, por partición, vista − no vista, ERA5-Land. Figuras: CDF, mapa, hidrogramas. **Falta:** el análisis cualitativo: por qué falla Magdalena-Cauca y el Pacífico (cuencas pequeñas e intervenidas, lluvia de CHIRPS, balance inconsistente; bitácora 8 y 13), qué pasa en los picos (FHV −28 %: subestima crecientes) y B7/B8 si alcanzan. Cada figura con 2–4 líneas en el texto |
| **Discusión y aportes (15 %):** por qué se obtuvieron los resultados, aportes, limitaciones (datos, sesgos, validez, generalización), aplicaciones, **ética o interpretabilidad** | ⚠️ | **Falta escribir:** (1) por qué 0,28 y no 0,69 como Kratzert 2019: 248 cuencas frente a 531, forzantes más pobres en el trópico, regiones con pocas cuencas; (2) ERA5-Land mejora la cuenca vista y no la no vista: más entradas, más ajuste a las cuencas vistas; (3) LSTM empata con LightGBM: con 248 cuencas, la ventaja secuencial no alcanza a notarse; (4) limitaciones: 95 cuencas retenidas, sesgo hacia el Caribe y los Andes, intervención humana; (5) interpretabilidad: importancia por permutación (bitácora 20); (6) ética: riesgo de usar la predicción para decisiones de riesgo sin estación, por eso la app muestra la confianza por región y el aviso |
| **Conclusiones y trabajo futuro (10 %):** hallazgos frente a cada objetivo; líneas concretas desde las limitaciones | ⚠️ | **Falta:** una conclusión por objetivo (cumplido o no, con el número). Trabajo futuro ya identificado (fase D): Caravan y CAMELS-PE por similitud, híbridos δ, AlphaEarth, cuencas anidadas, polígonos arbitrarios en la app |
| **Estructura y redacción (10 %):** título, resumen (200–300 palabras) y 3–5 palabras clave, introducción, metodología, resultados, discusión, conclusiones y referencias en formato consistente; redacción sin errores; **máximo 12 páginas con referencias**; la declaración de uso de IAG y los anexos no cuentan | ❌ | El `.tex` tiene la estructura de la propuesta (4 páginas). **Falta:** reestructurar con estos apartados, IEEE como estilo único de citas (la plantilla ya lo es), un enlace al repositorio en un anexo y la **declaración de uso de IA generativa** (se usó Claude Code para código, revisión de literatura y redacción). El enunciado dice "no más de 10 páginas" y la rúbrica "no excede 12": la meta es 10 |

### 10.2 Despliegue (25 %)

| Criterio (peso) | Estado | Plan |
|---|---|---|
| **Funcionalidad (50 %):** recibe entradas, usa el modelo entrenado, entrega resultados sin errores | ⚠️ | El visor de Sebastián ya recibe entradas y entrega resultados, pero solo con la línea base y fuera del repositorio. Fase C, tareas C0 a C3: subirlo, cargarle el LSTM de las dos etapas e inferencia real (ONNX o API) con el escenario de lluvia |
| **Interfaz para el usuario final (50 %):** lenguaje del contexto, resultados comprensibles, auto contenida | ⚠️ casi | El visor ya habla en m³/s, explica la confianza en palabras y tiene la pestaña "Acerca del modelo". Falta actualizar las metas y los números. Resto del plan original: | Fase C: m³/s y no mm/día, confianza en palabras por región, escenario de El Niño, texto de "qué hace y qué no hace", ayudas en cada control. Nada de "NSE" sin explicar en la pantalla principal |

### 10.3 Presentación (15 %)

| Criterio (peso) | Estado | Plan |
|---|---|---|
| Problema (20 %) | ⚠️ | 2 min: estaciones escasas, El Niño 2026–27, por qué importa el caudal en un río sin estación |
| Solución (25 %) | ⚠️ | 3 min: datos, LSTM regional sin caudal, validación espacial por sistema de ríos; una diapositiva con el diagrama |
| Resultados y análisis (25 %) | ⚠️ | 3 min: CDF y mapa, la brecha vista − no vista y las limitaciones dichas de frente (meta no alcanzada) |
| Demo (20 %) | ❌ | 3 min con los casos de C4; video de respaldo |
| Presentación y tiempo (10 %) | ❌ | Ensayo cronometrado el 1/12; sin leer las diapositivas; `.pptx` en el padlet antes de las 5:30 pm |
