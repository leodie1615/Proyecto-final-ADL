# Bitácora de modelado: LSTM regional para caudal diario en CAMELS-COL

Registro de los pasos, decisiones y resultados del modelado, para el informe y la presentación final. Cada entrada dice qué se hizo, por qué, qué se encontró y dónde está la evidencia (sección del notebook o carpeta de `runs/`). El preprocesamiento está documentado en `proyecto-final-preprocesamiento.ipynb`; aquí empieza en la base de `data/processed/`.

**Notebook de modelado:** `proyecto-final-modelado.ipynb`. **Corridas:** `runs/` (excluida de git). **Metas de la propuesta:** NSE y KGE medianos ≥ 0,70 en prueba (flujo temporal) y NSE mediano ≥ 0,50 en cuencas no vistas (flujo espacial).

---

## Resumen del estado (28/11/2026)

- **27–28/11, etapa 2 (cuencas no vistas, sección 24):** en 95 cuencas retenidas por sistema de ríos, el LSTM con ERA5-Land logra un NSE mediano de 0,300 (el base 0,276). Supera a la climatología de donantes (−0,08) y al vecino más cercano (−0,43), y empata con LightGBM (0,26). No llega a la meta de 0,50. Por no tener estación se pierde unos 0,14 de NSE por cuenca.

### Estado al 19/11/2026

| Modelo, prueba 2016–2022 (183 cuencas, mismos días) | NSE mediano | KGE mediano | Cuencas con NSE < 0 |
|---|---|---|---|
| Climatología de la cuenca (baseline) | 0,18 | 0,21 | — |
| LSTM, lote 256, entrenado con 1981–2010 (sección 8) | 0,394 | 0,513 | 20 |
| LightGBM (sección 8.3) | 0,390 | 0,525 | 20 |
| Promedio LSTM + LightGBM (mejor resultado hasta el 17/11) | 0,410 | 0,518 | 20 |
| LSTM con ERA5-Land, ensamble de 3 semillas (sección 22) | 0,432 | 0,539 | 14 |
| **Final: LSTM + LightGBM, los dos con ERA5-Land (sección 22)** | **0,437** | **0,536** | 17 |
| Meta de la propuesta | 0,70 | 0,70 | — |

- **19/11:** con ERA5-Land, el modelo final sube el NSE a 0,437 (+0,020 por cuenca frente al promedio del 17/11; 127 de 183 cuencas mejoran; p = 1e-8). El KGE casi no cambia. Sigue lejos de la meta (sección 22).
- El flujo temporal queda lejos de la meta. Ningún cambio de entrenamiento movió el NSE: los límites están en los datos de entrada y en cuencas cuya relación lluvia-caudal cambió con los años.
- Por región, Orinoco cumple (NSE 0,71) y Amazonas se acerca (0,61); Magdalena-Cauca (108 de las 183 cuencas) se queda en 0,32.
- El flujo espacial (5 particiones) está implementado pero no se ha entrenado.
- **17/11:**
  - Se descartó la segunda copia de CHIRPS (sección 12).
  - El balance de agua es inconsistente en 32 cuencas, pero sin ellas la mediana solo sube a 0,43 (sección 13).
  - En un piloto, la lluvia de ERA5 sumó apenas +0,01 de NSE (sección 14).
  - Las variables de cada iteración quedan en el registro del final y en `docs/registro-variables.csv`.
  - El catálogo de variables candidatas, con prioridades, está en la sección 16 y en `docs/variables-candidatas.csv`.
- **18/11:**
  - ERA5-Land ya está en series diarias por cuenca hasta 2017 (sección 17) y se validó contra las grillas originales y otras fuentes (sección 18). La descarga de 2018 a 2022 sigue en curso.
  - **Selección de variables con LightGBM, en validación 2011–2015** (sección 19): agregar ERA5-Land completo sube el NSE mediano de 0,446 a 0,480 (+0,020 por cuenca; 158 de 195 cuencas mejoran; 3 semillas). Es la primera fuente nueva que mueve el resultado. `eroded_misce_perc` se puede quitar. El Niño, el rango térmico, latitud y longitud y la estacionalidad no suman.
  - **Permutación en el LSTM** (sección 20): el LSTM depende de `pr` y, mucho más que LightGBM, de los atributos. Tampoco usa `eroded_misce_perc`.
  - Confirmación con el LSTM lanzada a las 22:32 (sección 21): 6 corridas (2 configuraciones × 3 semillas × 3 épocas fijas).
- **19/11:**
  - **La confirmación con el LSTM sostiene la mejora** (sección 21): con ERA5-Land el NSE mediano de validación sube de 0,452 a 0,491 (+0,024 por cuenca; 145 de 195 cuencas mejoran; p = 2e-12), en las 3 semillas y en todas las regiones. Coincide con LightGBM (+0,020).
  - La descarga de ERA5-Land terminó a la 01:10 (126 de 126 pedidos, sin fallos, 31,5 h). Se procesó 2018–2022: la tabla diaria y la base del LSTM (`data/processed_era5land`) quedaron completas de 1981 a 2022, sin huecos (sección 21).
  - **Evaluación final en la prueba 2016–2022** (sección 22, fijada antes de mirarla): el promedio del LSTM y LightGBM, los dos con ERA5-Land, da NSE 0,437 y KGE 0,536, frente a 0,410 y 0,518 del mejor anterior. ERA5-Land mejora la prueba lo mismo que la validación (+0,024 en el LSTM). Con ERA5-Land, el LSTM solo (0,432) ya casi iguala al promedio.
  - **Las pruebas pasaron a los notebooks, con sus salidas como evidencia.**
    - **Preprocesamiento, sección 2:** ERA5-Land (descarga, pesos, series diarias, validación y base del LSTM).
    - **Modelado, secciones 8.4 a 8.7:** selección con LightGBM, permutación en el LSTM, confirmación y evaluación final.
    - **Comprobación:** el código de los notebooks reproduce exactamente los archivos y resultados de los scripts:
      - los pesos, los promedios por cuenca, la tabla diaria y los bloques, bit a bit;
      - los resúmenes de 8.4, 8.6 y 8.7;
      - las configuraciones de las corridas.
    - Las celdas que entrenan o descargan tienen su bandera `EJECUTAR_*` en `False`; con la bandera así, muestran lo guardado.
  - **20/11: se reorganizó el notebook de modelado** en 11 secciones, con texto conciso y solo las salidas que aportan.
    - Equivalencias con los números que usa esta bitácora: 4 → 3, 5 → 4, 6 → 5, 8 → 6.1, 8.3 → 6.2, 8.4 a 8.7 → 6.4 a 6.7, 9 → 7, 10 → 8, 11 → 9 y 10, 12 → 11.
    - Se quitaron la prueba de humo (3), el ajuste de `hidden_size` (7), el lote 64 (8.1) y el entrenamiento con 1981–2015 (8.2). Quedan resumidos en las secciones 2.2, 2.3 y 6.
    - Nuevo: 6.3, las variables de ERA5-Land y su relación con las que ya había.
    - Los resultados de cada etapa quedaron en las secciones 9 y 10; la 9 incluye la climatología.
  - **Catálogo de alternativas para mejorar el modelo y exploración de transfer learning y fine-tuning** (sección 23; `docs/alternativas-mejora.csv`). Ninguna promete llegar a 0,70. La apuesta más barata con evidencia es el fine-tuning por cuenca; el transfer learning desde Caravan tiene más sentido para el flujo espacial.

---

## 1. Configuración base del modelo (sección 2 del notebook)

**Decisión.** Partir de la configuración validada en la literatura y no hacer una búsqueda amplia de hiperparámetros, por el presupuesto de cómputo (una GPU de 6 GB).

| Parámetro | Valor | Origen |
|---|---|---|
| Modelo | LSTM de una capa (`cudalstm`, NeuralHydrology 1.13) | Propuesta |
| Entradas | 4 forzantes (lluvia CHIRPS; evapotranspiración y temperaturas MSWX) + 48 atributos estáticos, sin caudal | Propuesta; preprocesamiento |
| Ventana | 365 días | Wi et al. (2025); Chao et al. (2026) |
| `hidden_size` | 256 unidades (317 697 parámetros) | Wi et al. (2025) |
| Dropout de salida | 0,4 | Wi et al. (2025) |
| Optimizador, tasa | Adam, 5 × 10⁻⁴ | Wi et al. (2025) |
| Pérdida | NSE* (error de cada cuenca normalizado por su desviación estándar) | Propuesta; Yu y Tolson (2026) |
| Épocas | Máximo 30, con parada temprana (paciencia 5, contada desde la época 6) | Propuesta |
| Semilla | 42 | Reproducibilidad |

---

## 2. Prueba de humo y reproducibilidad (10 y 11/11; sección 3)

- **Qué:** 20 cuencas (4 por región), 2 épocas.
- **Resultado:** la pérdida baja, se calculan las métricas y se guardan los checkpoints. Con lote 64: 2 min 20 s por época, NSE de validación 0,46.
- **Reproducibilidad:** al repetir la prueba (11/11), los resultados fueron idénticos hasta el quinto decimal. Con `seed: 42`, cada corrida se puede repetir exactamente.

## 3. Decisión: tamaño de lote 256 (11/11; secciones 3.1 y 8.1)

- **Por qué:** con lote 64, el plan completo (9 corridas) tomaba entre 51 y 139 horas de GPU. Con 256, cada época es entre 2,4 y 2,8 veces más rápida.
- **Riesgo aceptado:** con 4 veces menos actualizaciones por época, el modelo avanza menos en cada época. Se acordó volver a 64 si el resultado no alcanzaba la meta.
- **Comprobación (12/11):** el flujo temporal con lote 64 dio exactamente el mismo NSE de prueba (0,394) y tardó 1,5 veces más por época. **Se mantiene 256.**

## 4. Decisión: `hidden_size` fijo en 256 (11/11; sección 7)

- **Qué es:** el número de unidades (neuronas con memoria) de la capa oculta del LSTM.
- **Decisión:** usar 256, el valor de la propuesta, y dejar el ajuste (64, 128 y 256) como etapa opcional. Ahorra entre 5 y 14 horas de GPU.

## 5. Anidamiento y particiones espaciales (sección 5)

- **Problema:** si una cuenca está aguas arriba o aguas abajo de otra, retener una y entrenar con la otra filtra su caudal al modelo.
- **Método:** una cuenca está anidada en otra si su punto de aforo cae dentro del polígono de la otra. Se verificó con áreas.
- **Resultado:**
  - Las 248 cuencas forman 35 sistemas de ríos.
  - Los dos más grandes, Magdalena-Cauca (130) y Orinoco (23), siempre entrenan: son 153 cuencas fijas.
  - Las otras 95 se reparten en 5 particiones de 19.
- **Diferencia con la propuesta:** la propuesta hablaba de 216 fijas y 130 retenidas porque partía de 346 cuencas. El filtro de calidad del preprocesamiento dejó 248.

## 6. Errores de NeuralHydrology 1.13 encontrados y corregidos (11/11; secciones 2.3 y 6)

| Problema | Consecuencia | Solución |
|---|---|---|
| `early_stopping: True` también activa la reducción automática de la tasa de aprendizaje y exige dos parámetros más | El `base.yml` original fallaba al arrancar | Paciencia de 30 (el máximo de épocas): la tasa queda constante |
| La parada temprana conserva los pesos de la última época, no los de la mejor | Se habría evaluado un modelo peor | Se evalúa siempre la época de menor pérdida de validación |
| El registro (`logging.basicConfig`) solo se configura una vez por kernel | Desde la segunda corrida, todo se escribía en el registro de la primera | `reiniciar_registro()` antes y después de cada corrida |
| Una cuenca con un solo día válido rompe el cálculo de métricas (FMS/FLV) | La validación fallaba | Solo se validan y evalúan cuencas con al menos 365 días evaluables |

## 7. Flujo temporal, primera corrida (11/11; sección 8)

- **Corrida:** `runs/camels_col_temporal_h256_eaa981_1111_171248`.
- **Entrenamiento:** 248 cuencas, 1981–2010; unos 17 minutos por época. La parada temprana lo detuvo en la época 14; la mejor fue la 3.
- **Prueba (2016–2022, 183 cuencas):** NSE mediano 0,394, KGE 0,513 y RMSE 1,69 mm/día; 20 cuencas con NSE negativo. Supera la climatología (NSE 0,18) pero queda lejos de 0,70.
- **Picos subestimados:** FHV de −30 %, como en los tres papers de referencia.

## 8. Diagnóstico del flujo temporal (12 y 13/11)

| Hallazgo | Evidencia | Implicación |
|---|---|---|
| El modelo no ajusta bien ni los años que ya vio | NSE con la época 3: 0,55 en entrenamiento, 0,46 en validación y 0,39 en prueba | El límite no es solo sobreajuste; está en la información de entrada |
| Gran diferencia entre regiones | NSE: Orinoco 0,71, Amazonas 0,61, Pacífico 0,42, Caribe 0,32, Magdalena-Cauca 0,32 | Magdalena-Cauca (59 % de las cuencas de prueba) arrastra la mediana |
| En 50 cuencas cambió la relación lluvia-caudal entre 1981–2010 y 2016–2022 | El error de volumen del LSTM sigue ese cambio (correlación de Spearman −0,82); esas 50 cuencas tienen NSE 0,20 y las 133 estables, 0,41 | Probable intervención humana (embalses, captaciones) o cambios en la medición. Ningún modelo basado solo en el clima lo puede predecir |
| Los errores son sistemáticos | Promediar los dos LSTM (lote 256 y 64) solo sube el NSE de 0,394 a 0,395 | Los LSTM se equivocan igual en las mismas cuencas |
| La ventana de 365 días pierde datos | Las forzantes faltan los mismos días que el caudal; el 20 % de los días con caudal no tiene la ventana completa | Se pierden el primer año después de cada hueco y días de evaluación |

## 9. Intentos de mejora del flujo temporal

| Fecha | Intento | Resultado en prueba | Conclusión |
|---|---|---|---|
| 12/11 | Lote 64 (8.1) | NSE 0,394 (igual) | El lote no limita |
| 12/11 | Entrenar con 1981–2015 y 3 épocas fijas (8.2) | NSE 0,398; Wilcoxon p = 0,92 | Sin mejora significativa |
| 16/11 | **LightGBM** con promedios de lluvia de 1 a 365 días e índices de lluvia antecedente (8.3) | NSE 0,390 solo; **0,410 promediado con el LSTM** (mejoran 127 de 183 cuencas, p = 2 × 10⁻⁷) | Primera mejora significativa: LightGBM se equivoca distinto al LSTM. Entrena en 5 minutos en CPU |
| 16 y 17/11 | Rellenar los huecos de lluvia con CHIRPS y los de evapotranspiración y temperaturas con su promedio por día del año | LSTM 0,380 (p = 0,39), LightGBM 0,392, promedio 0,404 (p = 0,26) | Sin mejora. Sí permite simular el 100 % de los días: 187 cuencas, 14 con NSE negativo |

**Lecciones:**
- Dos modelos muy distintos llegan al mismo techo (cerca de 0,40). El límite está en los datos.
- Tener un 21 % más de ventanas de entrenamiento no mejoró el LSTM: importa más la calidad de la información que la cantidad de días.
- La lluvia de CAMELS-COL ya es CHIRPS. Lo dice la propuesta (sección V) y se confirmó con correlación diaria mediana de 0,987 en las 248 cuencas. CHIRPS no aporta información nueva, solo completa huecos.

**Cómo se hizo el relleno con CHIRPS (16 y 17/11):**

| Aspecto | Detalle |
|---|---|
| Variable | Solo lluvia diaria (mm/día). CHIRPS no tiene otras variables |
| Producto | CHIRPS v2.0 diario, celdas de 0,05° (unos 5,5 km), servido por ClimateSERV (NASA SERVIR), `datatype 0` |
| Promedio por cuenca | El servidor promedia (`operationtype 5`) las celdas que caen dentro del polígono de la cuenca |
| Polígonos | Shapefile de CAMELS-COL, campo `IDEAM_CODE` = `gauge_id`. Se pasaron de Mercator a latitud y longitud, se simplificaron con tolerancia de 0,001° (unos 110 m) y se usaron solo los anillos exteriores |
| Período | 1981-01-01 a 2022-12-31, pedido en 5 tramos de hasta 10 años, porque el servidor rechaza períodos más largos |
| Resultado | 248 cuencas, 15 340 días cada una, sin días vacíos. Correlación diaria con la lluvia de CAMELS-COL: mediana 0,987, mínima 0,898 |
| Uso en `data/processed_relleno/` | La lluvia es CHIRPS **en todos los días**, no solo en los huecos, para que la serie sea homogénea |
| Otras forzantes | Los huecos de evapotranspiración y temperaturas se rellenaron con el promedio de ese día del año en la cuenca, calculado solo con 1981–2010 y suavizado con una media móvil de 31 días. En total, 804 453 días (21,1 %) |
| Caudal | Sin cambios: nunca se rellena |

**Error encontrado y corregido (17/11).**
- **El error:** los polígonos de CAMELS-COL vienen en "WGS 1984 World Mercator", la versión elipsoidal (EPSG:3395), según el `.prj`. Para pasarlos a latitud y longitud se usó la fórmula esférica (Web Mercator). Los polígonos enviados a ClimateSERV quedaron corridos hacia el ecuador entre 1,5 km (2° N) y unos 8 km (11° N).
- **La prueba:** la coincidencia entre CHIRPS y la lluvia de CAMELS-COL empeora en las cuencas pequeñas del norte. La correlación diaria mediana va de 0,995 (grandes, cerca del ecuador) a 0,968 (menores de 300 km², al norte de 6°). La correlación de Spearman entre el desfase relativo y la coincidencia es −0,71.
- **Consecuencia:** las conclusiones se mantienen. Con los polígonos bien ubicados, CHIRPS coincidiría todavía más con nuestra lluvia, y sería aún más redundante.
- **Corrección:** la conversión ya usa la fórmula elipsoidal, verificada con ida y vuelta exacta. Cualquier extracción futura (ERA5-Land) parte de ella. No se volvió a descargar CHIRPS.

**Dónde:** secciones 8.1 a 8.3 del notebook. La exploración con CHIRPS se hizo con scripts aparte; los datos están en `data/externos/chirps_climateserv/` y `data/processed_relleno/`, y la corrida en `runs/camels_col_temporal_relleno_1711_045443`.

## 10. ¿Qué aportaría ERA5-Land? (17/11)

**Información nueva que traería:**
- **Una lluvia de otra fuente.** ERA5-Land toma la lluvia del reanálisis atmosférico ERA5 y la interpola a 0,1° (unos 11 km). Sus errores son distintos a los de CHIRPS, que combina satélite infrarrojo y estaciones.
- **Variables que no tenemos:**
  - humedad del suelo en 4 capas (el estado de almacenamiento de la cuenca);
  - radiación solar, humedad del aire y viento;
  - evaporación real.

  Wi et al. (2025) usan 9 variables de clima, incluidas radiación y humedad.
- **Es la fuente de las forzantes de Caravan**, así que abriría la puerta al transfer learning (sección 11).

**¿Para qué cuencas?** Se clasificaron las 183 cuencas de prueba según por qué falla el mejor modelo (promedio LSTM + LightGBM). La tabla por cuenca está en `docs/diagnostico-cuencas.csv`.

| Grupo | Criterio | Cuencas | NSE mediano | Dónde | ¿Ayudaría ERA5-Land? |
|---|---|---|---|---|---|
| 1. La lluvia no alcanza | Caudal ≥ 90 % de la lluvia en 1981–2010 | 9 | 0,30 | Caribe (Atrato) 4, Pacífico 3 | **Sí, es el caso más claro** |
| 2. Cambió la relación lluvia-caudal | Cambio de más de 20 % entre 1981–2010 y 2016–2022 | 48 | 0,22 | Magdalena-Cauca 28 | No: el problema no es el clima |
| 3. Mala sincronía | Correlación entre simulado y observado < 0,6 | 27 | 0,21 | Magdalena-Cauca 20; cuencas pequeñas (336 km²) y altas (1 900 m) | Quizás, como segunda fuente |
| 4. Resto | — | 82 | 0,46 | — | Algo |
| 5. Ya cumple | NSE ≥ 0,7 | 17 | 0,83 | Orinoco 13; cuencas grandes y bajas | No hace falta |

**Comparación de la lluvia de ERA5 y CHIRPS** (2001–2010, 23 cuencas de los grupos 1, 3 y 5):
- **Fuente:** ERA5 vía Open-Meteo, sin cuenta, promediando hasta 6 puntos de su grilla dentro de cada cuenca.
- **Tabla por cuenca:** `docs/era5-vs-chirps-2001-2010.csv`.

| Grupo | Correlación diaria ERA5-CHIRPS | Correlación en sumas de 5 días | Caudal ÷ lluvia, con CHIRPS | Caudal ÷ lluvia, con ERA5 |
|---|---|---|---|---|
| 1. La lluvia no alcanza | 0,33 | 0,48 | 0,91 | 0,66 |
| 3. Mala sincronía | 0,37 | 0,60 | 0,50 | 0,38 |
| 5. Ya cumple | 0,52 | 0,73 | 0,58 | 0,68 |

**Conclusiones:**
- **ERA5 sí es información nueva:** su correlación diaria con CHIRPS es baja, de 0,33 a 0,52. CHIRPS solo repetía la lluvia de CAMELS-COL.
- **Grupo 1 (Atrato y Pacífico):** en 7 de 9 cuencas, ERA5 tiene bastante más lluvia. Por ejemplo, 4 875 mm/año frente a 2 157. Con ERA5, la fracción de la lluvia que sale como caudal baja de valores casi imposibles (0,9 a 1,07) a 0,4–0,85. Coincide con la subestimación de CHIRPS que reporta el estudio de CAMELS-COL.
- **ERA5 tiene sus propios sesgos:**
  - en 2 cuencas del grupo 1 da menos lluvia que CHIRPS, y la fracción que sale como caudal pasa de 1,5;
  - en cuencas andinas pequeñas llega a casi 4 veces la lluvia de CHIRPS.

  Por eso no conviene reemplazar CHIRPS. Lo indicado es darle ambas al modelo como entradas, para que aprenda a combinarlas según la cuenca (Kratzert et al., 2021).

  **Actualización (17/11):** el piloto de la sección 14 lo midió. ERA5 sola empeora el NSE en 0,10, y como segunda lluvia lo mejora solo en 0,01. Además, los puntos de esta comparación se tomaron con los polígonos corridos por el error de la sección 9.
- **Comprobación con el balance de agua:** a largo plazo, lluvia = caudal + evaporación, y la evaporación real no puede superar mucho a la evapotranspiración potencial (unos 900 a 1 500 mm/año en estas cuencas).
  - **Grupo 1:** con CHIRPS, la evaporación implícita es casi nula o negativa (−118 a 918 mm/año), así que CHIRPS se queda corta. Con nuestra propia lluvia de 1981–2010 pasa lo mismo: el caudal es entre el 91 % y el 104 % de la lluvia, y la evaporación implícita va de −273 a 508 mm/año. Con ERA5 a veces se pasa: hasta 6 552 mm/año en el Atrato. La lluvia real estaría entre las dos.
  - **Grupo 3:** CHIRPS da evaporaciones creíbles, mientras que ERA5 implica hasta 6 597 mm/año: ERA5 sobreestima en las laderas andinas.
  - **Orinoco:** ERA5 se queda corta, con evaporación implícita cercana a 0.
- **No resuelve el grupo 2:** esas 48 cuencas no dependen de la calidad del clima.
- **Cómo se unen los datos de grilla a las cuencas.** ERA5-Land y CHIRPS son grillas de celdas sin identificador de cuenca. La unión es espacial:
  1. Se toma el polígono de cada cuenca del shapefile `CAMELS_COL_catchments_boundaries` (346 polígonos; su campo `IDEAM_CODE` es nuestro `gauge_id`).
  2. Se calcula qué fracción de cada celda cae dentro.
  3. Se promedian las celdas con esos pesos, día por día.
  4. El resultado es una tabla con `gauge_id`, fecha y variables, que se une a la base por esas dos columnas.

  Ejemplo en `docs/figuras/poligono-52017080.png`: la cuenca 52017080 (247 km²) cubre 7 celdas de CHIRPS y toca 8 de ERA5-Land, de las cuales 4 cubren al menos el 21 %.
- **Acceso:** Open-Meteo sirvió para esta prueba, pero no da la lluvia de ERA5-Land, tiene límites de uso y entrega puntos, no promedios por cuenca. Para todas las variables y cuencas hace falta una cuenta gratuita de Copernicus CDS o de Google Earth Engine.

## 11. Revisión de los papers de referencia (17/11)

**Qué:** buscar en los trabajos citados en la propuesta (referencias [5] a [12]) qué decisiones toman al usar un LSTM para caudal diario, y si las podemos aplicar.

**Fuentes leídas:**
- **Texto completo:**
  - Wi et al. (2025), copia en el repositorio de la NSF.
  - Yu y Tolson (2026), capítulo 3 de la tesis doctoral de Yu, Universidad de Waterloo.
  - Kapoor y Chandra (2026), preprint en arXiv.
- **Presentación de los autores:** Chao et al. (2026).
- **Solo el resumen:** Bakar et al. (2026).
- **Sin acceso al texto:** He et al. (2024), Barbhuiya y Gupta (2025) y Alzhanov et al. (2025). Se usa el resumen de la propuesta.

| Consideración | Qué hacen los papers | Qué hacemos | ¿La implementamos? |
|---|---|---|---|
| **Ensamble de semillas** | Wi et al. entrenan cada modelo 10 veces con distinta inicialización y promedian. Yu y Tolson también promedian 10 semillas. Chao et al. reportan media ± desviación de varias corridas | Una sola semilla (42). Nuestro promedio de dos LSTM no mejoró, pero ambos usaban la misma semilla | **Sí, barato:** 5 semillas con 3 épocas fijas son unas 4 horas de GPU. Mejora esperada pequeña |
| **Separar cuencas naturales y reguladas** | Yu y Tolson usan solo estaciones de flujo natural. Wi et al. evalúan aparte las cuencas reguladas, que rinden peor | Todas juntas | **Sí, sin costo:** reportar aparte las 50 cuencas cuya relación lluvia-caudal cambió, como indicador de intervención |
| **Recencia de los datos** | Yu y Tolson: los datos de más de unos 30 años antes de la prueba no ayudan. Si hay que recortar, mejor quitar años viejos que cuencas | Entrenamiento 1981–2010. Sumar 2011–2015 no mejoró | **Sí, barato:** entrenar con 1991–2015, en una corrida de alrededor de 1 hora |
| **Tasa de aprendizaje decreciente** | Yu y Tolson la ajustan por tramos, por ejemplo 0,001 → 0,0005 → 0,0001 cada 10 épocas | Constante en 5 × 10⁻⁴ (por el error de NeuralHydrology, sección 6) | **Sí:** una corrida. Nuestra mejor época siempre es la 3, señal de que la tasa podría ser alta |
| **Ajuste de hiperparámetros por validación cruzada** | Wi et al. y Yu y Tolson ajustan tamaño (128 a 512), lote, ventana (30, 180 o 365 días), dropout (0,2 a 0,4) y épocas con 5 particiones | Fijos, por presupuesto | Parcial: el lote ya se probó sin efecto. Se podrían probar 128 unidades y una ventana de 180 días |
| **Más variables de clima** | Wi et al. usan 9 de un reanálisis: lluvia, temperaturas, radiación, humedad, presión, viento y evapotranspiración | 4 | Requiere ERA5-Land (sección 10) |
| **Transfer learning desde Caravan** | Alzhanov et al. preentrenan con Caravan (miles de cuencas) y ajustan con su cuenca, con beneficio limitado según la propuesta | No | Requiere usar las mismas forzantes de Caravan, que son de ERA5-Land. Esfuerzo alto |
| **Híbrido con un modelo hidrológico** | Kapoor y Chandra agregan variables de GR4J y mejoran el NSE entre 0,06 y 0,09 por estado, en modelos por cuenca. Chao et al. usan la salida de un modelo de procesos como entrada y bajan el sesgo de picos (FHV) de −10,9 % a −5,5 % | No | Posible si alcanza el tiempo: calibrar GR4J por cuenca (4 parámetros) y agregar sus salidas como entradas. Esfuerzo medio-alto |
| **Mezcla de expertos** | Chao et al.: un LSTM para caudales comunes y otro para extremos. NSE de 0,847 a 0,859 | No | Requiere programar un modelo propio en NeuralHydrology. No es prioridad |
| **Pérdida cuantílica e incertidumbre** | Kapoor y Chandra: un modelo por cuantil (5 %, 50 % y 95 %), con intervalos del 90 % | Predicción puntual | Opcional: NeuralHydrology tiene salidas probabilísticas. Aporta incertidumbre, no NSE |
| **Transformers** | Bakar et al.: en 671 cuencas de Estados Unidos, el Temporal Fusion Transformer llega a KGE 0,87, frente a 0,81 del LSTM con atributos y 0,78 del LSTM sin ellos | LSTM con 48 atributos | Poco probable que ayude con nuestro límite de datos |
| **Cuencas donantes con caudal observado** | Wi et al.: su mejor modelo para cuencas no aforadas usa el caudal de 5 estaciones vecinas | No | Choca con la regla del proyecto de no usar caudal observado como entrada. Solo aplicaría al flujo espacial; lo decide el equipo |
| **Ventana de 365 días; 256 unidades, dropout 0,4, tasa 5 × 10⁻⁴, lote 64** | Wi et al., elegidos con validación cruzada. Chao et al. usan 365 días | Igual, con lote 256 | Hecho |
| **Atributos estáticos** | Wi et al. usan 30, Yu y Tolson 16, Chao et al. elevación, suelo, cobertura y longitud del cauce. Bakar et al.: los atributos mejoran el KGE de 0,78 a 0,81 | 48 | Hecho |
| **Modelo regional único** | He et al.: supera a modelos por clase de cuenca cuando hay pocas cuencas por clase | Regional | Hecho |
| **Excluir el caudal faltante de la pérdida** | Barbhuiya y Gupta | Igual | Hecho |
| **Descartar cuencas con muchos faltantes** | Wi et al. exigen menos de 5 % de faltantes; Yu y Tolson hasta 10 %; Kapoor y Chandra descartan las de más de 10 % | No se descartan: el umbral de 5 % eliminaría 136 de las 248 cuencas | Decisión del preprocesamiento; se mantiene |

**Prioridades para el flujo temporal:**
1. **Separar en la evaluación** las cuencas cuya relación lluvia-caudal cambió. No cuesta nada y explica buena parte del resultado.
2. **Ensamble de 5 semillas** del LSTM, combinado con LightGBM. Unas 4 horas de GPU.
3. **Recencia:** entrenar con 1991–2015. Alrededor de 1 hora.
4. **Tasa de aprendizaje decreciente.** Una corrida.
5. **ERA5-Land**, con transfer learning desde Caravan como extensión. Necesita una cuenta de Copernicus o Earth Engine. Tiene el mayor potencial y el mayor esfuerzo.
6. **Híbrido con GR4J**, si alcanza el tiempo.

**Referencias consultadas:**
- Wi, Gupta y Steinschneider (2025), copia en el repositorio de la NSF: https://par.nsf.gov/servlets/purl/10648170
- Yu (2026), tesis doctoral, Universidad de Waterloo, capítulo 3 (= Yu y Tolson, 2026): https://uwspace.uwaterloo.ca/items/c257acc7-7e23-4ab6-8465-22889fd78717
- Kapoor y Chandra (2026), preprint: https://arxiv.org/abs/2510.05453
- Chao y Arnold (2025), presentación en CWEMF: https://cwemf.org/wp/wp-content/uploads/2025/11/16.4_DailyStreamflowDeepLearning_ChaoHe.pdf
- Bakar et al. (2026), resumen: https://www.rti.org/publication/deep-learning-approaches-for-streamflow-flash-drought-prediction-

## 12. Se descarta la lluvia de ClimateSERV; cómo se calcula la lluvia de una cuenca (17/11)

**Decisión:** se descartan la lluvia de CHIRPS bajada de ClimateSERV y la base rellenada (`data/processed_relleno/`, iteraciones T6 a T8).
- **Razones:** es el mismo producto que nuestra `pr` (correlación diaria de 0,987, misma media), y rellenar con ella no mejoró el resultado.
- **Qué se conserva:** nuestra `pr` sigue siendo CHIRPS. Lo que se descarta es la segunda copia.
- **Los archivos:** ningún notebook los usa. Quedan en `data/`, fuera de git, por si sirven para evaluar más días en el flujo espacial.

**Las grillas de lluvia**

| | CHIRPS v2.0 | ERA5 | ERA5-Land |
|---|---|---|---|
| Quién la produce | Climate Hazards Center (Universidad de California en Santa Bárbara) | ECMWF, para Copernicus | ECMWF, para Copernicus |
| De dónde sale la lluvia | Satélite infrarrojo (duración de nubes frías), calibrado con una climatología y mezclado con estaciones | Reanálisis: un modelo del tiempo que asimila observaciones. La lluvia la calcula el modelo; no es una medición | La misma lluvia de ERA5, interpolada a 0,1°. Lo nuevo son las variables de suelo, que calcula un modelo de superficie alimentado con ERA5 |
| Tamaño de celda | 0,05°: unos 5,5 km de lado (31 km²) | 0,25°: unos 28 km (770 km²) | 0,1°: unos 11 km (124 km²) |
| Paso de tiempo y período | Diario, desde 1981 | Horario, desde 1940 | Horario, desde 1950 |
| Variables | Solo lluvia | Cientos, de atmósfera y superficie | Unas 50 de superficie: humedad del suelo en 4 capas, evaporación, escorrentía, temperatura, radiación, humedad del aire, viento |
| Cobertura | Tierra entre 50° S y 50° N | Global | Global, solo tierra |
| Acceso | ClimateSERV o archivos de la UCSB, sin cuenta | CDS, con cuenta; Open-Meteo sin cuenta, por puntos | CDS, con cuenta; Google Earth Engine |

**Cómo se calcula la lluvia de una cuenca con una grilla.** Ejemplo: cuenca 52017080, en `docs/figuras/poligono-52017080.png`. La figura ahora tiene dos paneles, uno por grilla; antes estaban superpuestas y se confundían.
- **La línea negra no es el río:** es el límite de la cuenca, la divisoria de aguas que separa lo que drena hacia la estación de lo que drena hacia otros ríos. El río va por dentro, y su cauce principal mide 34,9 km.
  - Por eso la línea es cerrada.
  - La estación (triángulo naranja, agregado el 17/11) está en el punto más bajo del límite, la salida de la cuenca, a 599 m; lo más alto está a 3490 m.
- **Cada celda trae un solo valor por día:** la lluvia promedio de toda la celda. No se sabe en qué parte de la celda llovió, así que se supone pareja dentro de ella.
- **Qué significa el porcentaje:** se mide qué fracción del área de cada celda queda dentro de la cuenca (f). El "21 %" del ejemplo es el 21 % de la celda, no de la cuenca.
- **La fórmula:** lluvia de la cuenca = Σ (f × P) / Σ f. El peso de cada celda es f / Σ f; la celda del 21 % pesa 0,212 / 2,002 = 10,6 %.
- **Ejemplo con lluvias inventadas:**
  - Celdas con f = 0,77, 0,57, 0,35, 0,21, 0,09 y 0,01, y lluvias de 12, 8, 20, 5, 30 y 40 mm.
  - Resultado: 24,87 / 2,00 = 12,4 mm.
  - Con la regla del centro, que solo cuenta las 2 celdas cuyo centro cae dentro (12 y 8 mm), daría 10,0 mm.
- **Comprobación:** Σ f × área de la celda = 2,00 × 123,8 km² = 248 km², que es el área de la cuenca (247 km²). Con CHIRPS: 8,01 × 31,0 km² = 248 km².
- **La regla de CAMELS-COL:**
  - Según su descripción, la lluvia de la cuenca es el "promedio aritmético de todos los píxeles dentro de sus límites". La forma habitual de aplicarlo es contar las celdas cuyo centro cae dentro: en el ejemplo, 7 de las 16 que tocan la cuenca, todas con el mismo peso.
  - Con celdas pequeñas funciona bien. Con celdas grandes falla: en ERA5-Land solo 2 celdas tienen el centro dentro, y una cuenca muy pequeña podría no tener ninguna.
  - Por eso, para ERA5-Land, se propone el promedio ponderado.

## 13. Balance de agua: ¿cuánto pesa la inconsistencia? (17/11)

A largo plazo, la lluvia es igual al caudal más la evaporación real, y la evaporación real debe estar entre 0 y la potencial. Se revisó en las 248 cuencas con todos los días que tienen caudal y lluvia observados (1981–2022).

| Inconsistencia | Criterio | Cuencas | % de los días con caudal | Dónde |
|---|---|---|---|---|
| Más caudal que lluvia | Q/P > 1 | 0 | 0 % | El pre-filtrado ya había descartado 55 cuencas así (bandera `f_q_gt_p`). Su coeficiente de escorrentía publicado va de 1,01 en adelante (mediana 1,27); el máximo entre las conservadas es 0,99. El criterio fue Q/P > 1 |
| La lluvia no alcanza | Q/P ≥ 0,9: se evaporaría menos del 10 % de la lluvia | 18 (7,3 %) | 6,6 % | En todas las regiones: Caribe 5, Orinoco 4, Amazonas 3, Pacífico 3 y Magdalena-Cauca 3 |
| Sobra lluvia, o sale agua sin pasar por la estación | Evaporación implícita (P − Q) mayor que 1,2 veces la potencial | 14 (5,6 %) | 6,2 % | Cuencas andinas pequeñas: Caribe 7 y Magdalena-Cauca 7 |
| **Total** | | **32 (12,9 %)** | **12,7 %** | |

- **Escala anual:** el 3,9 % de los años-cuenca tiene más caudal que lluvia. En un año puntual puede pasar, porque el suelo y los acuíferos guardan agua de un año a otro.
- **Escala diaria:** no aplica. En el 55 % de los días el caudal supera la lluvia de ese día, porque el río sigue llevando agua en los días secos.
- **Efecto en el resultado:** de las 183 cuencas de prueba, 22 son inconsistentes. Su NSE mediano es 0,22, frente a 0,43 del resto. Sin ellas, la mediana general pasaría de 0,41 a 0,43.
- **Conclusión:** es decisivo para esas 32 cuencas, pero no para la meta. Aun con datos consistentes, la mediana queda lejos de 0,70.
- **Propuesta:** marcarlas y reportarlas aparte, junto con las cuencas intervenidas (grupo 2).
- **Posible causa de los casos en que sobra lluvia:** de las 11 que están en prueba, 6 son del grupo 2 (cambió la relación lluvia-caudal) y 3 del grupo 3. Encaja con extracciones de agua o trasvases aguas arriba de la estación, aunque también puede ser que CHIRPS sobreestime la lluvia en las laderas.

## 14. Piloto: ¿la lluvia de ERA5 le aporta al modelo? (17/11)

**Pregunta:** si el modelo recibe la lluvia de ERA5 además de la nuestra, ¿predice mejor el caudal?

**Diseño:**
- **Cuencas:** 30. Son las 23 de la comparación de la sección 10 (grupos 1, 3 y 5) y 7 tomadas al azar de los grupos 2 y 4. Se planearon 35, pero el límite diario de Open-Meteo cortó la descarga.
- **Lluvia de ERA5:** de Open-Meteo, 2001–2010, en celdas de 0,25°.
  - Se promedia según la fracción de cada celda dentro de la cuenca (sección 12), con los polígonos ya corregidos y el día local.
  - Está bien alineada en el tiempo: la correlación con nuestra lluvia es máxima sin desfase (0,38, frente a 0,26 y 0,22 corriéndola un día).
- **Modelo:** LightGBM con las mismas variables derivadas de T4 y los 48 atributos.
- **Validación:** cruzada por bloques de 2 años. Se entrena con 8 años y se predicen los 2 restantes, cinco veces, así que cada día se predice sin haberlo visto.
- **Entradas:** tres conjuntos, evaluados en los mismos días.

| Entradas | NSE mediano | Cuencas que mejoran frente a P1a | Cambio mediano | Wilcoxon |
|---|---|---|---|---|
| P1a. Nuestra lluvia (CHIRPS) | 0,490 | — | — | — |
| P1b. Nuestra lluvia + ERA5 | 0,493 | 19 de 30 | +0,009 | p = 0,006 |
| P1c. Solo ERA5 | 0,378 | 2 de 30 | −0,103 | p = 2 × 10⁻⁸ |
| Promedio de P1a y P1c | 0,489 | 10 de 30 | −0,013 | p = 0,18 |

- **Grupo 1 (la lluvia no alcanza):** pasa de 0,390 a 0,412. Las mayores ganancias: 11047010 (Atrato), de 0,20 a 0,32, y 52017080 (Pacífico), de 0,23 a 0,29.
- **Resultado por cuenca:** `docs/piloto-era5-nse.csv`. Datos en `data/externos/era5_openmeteo/`.

**Conclusiones:**
- **ERA5 sola es peor que CHIRPS** en 28 de 30 cuencas. CHIRPS es la mejor lluvia que tenemos.
- **Como segunda lluvia, ERA5 ayuda poco pero de forma consistente:** mejoran 19 de 30 cuencas y la diferencia es significativa. La ganancia típica es de 0,01 en NSE, y llega a 0,11 en una cuenca del Atrato.
- **No acerca a la meta de 0,70.**
- **Límites del piloto:**
  - solo 30 cuencas, más de la mitad de grupos con problemas;
  - un solo decenio;
  - LightGBM, no el LSTM;
  - ERA5 a 0,25°, con hasta 10 celdas por cuenca.

  La lluvia de ERA5-Land es la misma de ERA5 interpolada, así que el resultado debería mantenerse con ella.
- **Falta medir** la humedad del suelo, la radiación y la humedad del aire de ERA5-Land, que es lo único realmente nuevo que traería.

**¿Daría distinto con la prueba completa? (17/11)**
- **Incertidumbre del piloto:** con bootstrap sobre las 30 cuencas (20 000 remuestreos):
  - la ganancia media de agregar ERA5 es +0,012, con intervalo del 95 % entre +0,004 y +0,022;
  - la mediana es +0,010, con intervalo entre −0,003 y +0,015;
  - ninguna cuenca pierde más de 0,022.
- **El piloto sobrerrepresenta las cuencas donde más ayuda ERA5.** La ganancia media en el grupo 1 es +0,026, pero ese grupo es 9 de las 30 cuencas del piloto y solo 9 de las 183 de prueba. Reponderando los grupos a la composición de prueba, la ganancia media baja a unos +0,004. El grupo 2 queda casi sin medir, con 1 sola cuenca.
- **El LSTM podría sacar algo más que LightGBM:** recibe la lluvia diaria completa y aprende a combinar las dos según la cuenca. En CAMELS de EE. UU., darle varias lluvias subió la mediana unas pocas centésimas (Kratzert et al., 2021), y esas lluvias eran más parecidas entre sí.
- **Conclusión:** en la prueba completa se espera una ganancia de entre 0 y 0,03. No cambia la conclusión: ERA5 no acerca a 0,70, y su lluvia no justifica por sí sola la descarga.

## 15. Decisiones pendientes y próximos pasos

| Tema | Propuesta | Estado |
|---|---|---|
| Modelo oficial del flujo temporal | Promedio LSTM + LightGBM (NSE 0,410), con el LSTM solo como referencia | Por confirmar con el equipo |
| Cuencas intervenidas | Reportar aparte las 48 cuencas del grupo 2 (sección 10) | Por hacer, sin costo |
| Mejoras baratas | Ensamble de 5 semillas, entrenamiento con 1991–2015 y tasa de aprendizaje decreciente (sección 11) | Por decidir |
| ERA5-Land | Como segunda lluvia aporta poco: +0,01 en NSE (sección 14). Falta medir humedad del suelo y radiación; luego, transfer learning desde Caravan | Acceso listo el 17/11: cuenta, clave en `.cdsapirc`, licencia aceptada y `cdsapi` 0.7.7. La prueba de un mes (enero de 2010, 15 variables) sobre la caja de las 248 cuencas (lon −78,1 a −67,6; lat −1,7 a 11,4) tardó 3,6 minutos y pesó 22 MB. Proyección para 1981–2022: unos 11 GB y entre 8 y 30 horas de cola | **Descarga en curso desde el 17/11 por la tarde**, en `data/externos/era5land/` (registro en `descarga.log`). Pedidos por año, 3 a la vez; se reanuda sola si se corta. Por la poca RAM, mientras corre no se ejecuta nada más. Avance al 18/11, 07:10: 86 de 126 pedidos (1981–2009 completos, 7,6 GB) en 13,5 h. El ritmo bajó de unos 9 pedidos por hora en la tarde a unos 3 en la madrugada (cola del CDS, de 18 a 80–90 min por pedido). Hubo 7 cortes de conexión entre las 05:56 y las 06:15; la librería reanudó sola y los archivos afectados se leyeron completos (pasos de tiempo completos, 9,6 % de celdas vacías = mar, rangos físicos plausibles). El paso a series diarias por cuenca quedó listo el 18/11 y ya se corrió con 1981–2009 (sección 17). **Descarga completa el 19/11 a la 01:10** (126 de 126 pedidos, sin fallos, 31,5 h); procesado 1981–2022 sin huecos |
| Catálogo de variables candidatas | 51 variables con fuente, acceso y prioridad (sección 16) | Hecho el 17/11; el equipo elige cuáles probar |
| Cuencas con balance de agua inconsistente | Marcar y reportar aparte las 32 cuencas de la sección 13, junto con las del grupo 2 | Por hacer, sin costo |
| Selección de variables | Pruebas de ablación con LightGBM (quitar una variable o un grupo y reentrenar), guiadas por el registro de variables | **Filtro con LightGBM y permutación en el LSTM hechos el 18/11** (secciones 19 y 20): ERA5-Land completo suma +0,020 de NSE en validación; `eroded_misce_perc` sale. **Confirmado con el LSTM el 19/11** (sección 21): +0,024 por cuenca en validación. **Falta:** decidir cómo probar los atributos en cuencas no vistas |
| Flujo espacial (5 particiones) | Entrenar con lote 256: unas 20 horas para el LSTM y unos 25 minutos para LightGBM | Pendiente. Conviene hacerlo con la configuración C2 (con ERA5-Land, sección 21) |
| Alternativas de mejora | Catálogo ordenado y plan de transfer learning y fine-tuning en 3 fases (sección 23) | Propuesto el 19/11: fases 0 y 1, baratas. Decide el equipo |
| Exploración con CHIRPS | Se descarta la lluvia de ClimateSERV y la base rellenada (sección 12) | Decidido el 17/11 |

**Cómo mantener esta bitácora:** agregar una entrada por cada paso o decisión, con la fecha, qué se hizo, por qué, el resultado y dónde está la evidencia. Actualizar el resumen del inicio cuando cambie el mejor resultado.

## 16. Catálogo de variables candidatas (17/11)

**Qué:** antes de las pruebas de ablación, se reunieron todas las variables que podrían servir: las que ya usamos y las candidatas. **El catálogo completo, con 51 filas, está en `docs/variables-candidatas.csv`.** Cada fila tiene:
- la fuente, la resolución, el período y el acceso;
- cómo se lleva a la cuenca;
- qué aportaría, y qué grupos del diagnóstico de la sección 10 atacaría;
- el esfuerzo y la prioridad.

**Criterios de prioridad:**
- que ataque los grupos que más fallan: el grupo 1 (la lluvia no alcanza) y el grupo 2 (cambió la relación lluvia-caudal, 48 cuencas);
- que cubra todo 1981–2022, porque el LSTM necesita secuencias sin huecos;
- que sea barata de conseguir;
- que respete la regla de no usar caudal observado.

**Prioridad A**

| Variable | Fuente | Por qué | Costo |
|---|---|---|---|
| Rango diario de temperatura | `t_max − t_min` | Indicador de nubosidad y radiación | Cero |
| Latitud y longitud | CAMELS-COL | Patrones regionales | Cero |
| Índice de El Niño (ONI) | NOAA, sin cuenta | El Niño y La Niña mueven la lluvia y el almacenamiento | Casi cero |
| Humedad del suelo en 4 capas | ERA5-Land | Agua guardada en la cuenca; lo único realmente nuevo de ERA5-Land | Una descarga |
| Radiación solar y humedad del aire | ERA5-Land | Energía y sequedad del aire; Wi et al. (2025) las usan | En la misma descarga |
| Embalses aguas arriba, según su año de construcción | Global Dam Watch o GRanD | Regulación: grupo 2 | Bajo |
| Grado de regulación, riego, población, agua subterránea | HydroATLAS | Llena el hueco de intervención humana; son los atributos de Caravan | Medio |

**Prioridad B, que conviene probar después:**
- **Otras lluvias y lo que trae ERA5-Land:**
  - la lluvia MSWEP, la mejor candidata para el grupo 1;
  - la evaporación real y la escorrentía modelada de ERA5-Land;
  - la evapotranspiración de Penman-Monteith en lugar de Hargreaves.
- **Almacenamiento:** GLEAM4, que trae humedad de la zona de raíces y evaporación real.
- **Cobertura y uso del agua:**
  - la cobertura anual de MapBiomas de 1985 a 2024, para la deforestación del grupo 2;
  - la demanda de agua del Estudio Nacional del Agua;
  - el % de páramo.
- **Suelos y subsuelo:**
  - las propiedades físicas del suelo de SoilGrids, que podrían reemplazar los 13 órdenes de suelo;
  - la permeabilidad de GLHYMPS;
  - la profundidad hasta la roca;
  - la estacionalidad de la lluvia.

**Prohibidas por la regla del proyecto:**
- las firmas hidrológicas;
- el caudal de días anteriores o de cuencas vecinas;
- la salida de un modelo hidrológico como GR4J calibrado con nuestro caudal.

  Esto último corrige la sección 11: allí el híbrido con GR4J figuraba como posible. Un GR4J sin calibrar sería válido, pero débil.

**Método de selección propuesto (17/11, por confirmar con el equipo).** El objetivo es escoger las entradas del LSTM y, con ellas, las del promedio LSTM + LightGBM. LightGBM sirve de filtro barato y el LSTM confirma.
1. **Filtro con LightGBM:** unos 5 minutos por prueba, frente a 1 a 4 horas del LSTM.
   - **Ablación:** quitar una variable o un grupo y reentrenar.
   - **Adición:** agregar una candidata y reentrenar.

   En ambos casos se compara el NSE por cuenca con el modelo completo, en los mismos días.
   - **Se mide en validación (2011–2015), nunca en prueba.** Elegir variables mirando la prueba haría optimista el resultado final. Las comparaciones T1 a T8 se hicieron en prueba; desde aquí, la prueba se usa una sola vez, al final.
   - **Línea base con 3 semillas,** para conocer el ruido. Una diferencia menor que ese ruido no cuenta.
   - **Por grupos antes que variable por variable:** 6 grupos de atributos en lugar de 48 entrenamientos.
   - **Los atributos estáticos también se prueban dejando cuencas fuera** (las 5 particiones espaciales). En el flujo temporal un atributo puede servir solo para reconocer la cuenca, como se sospecha de `eroded_misce_perc`, y eso no se generaliza a cuencas nuevas.
2. **Importancia por permutación en el LSTM ya entrenado:** sin reentrenar, se revuelve una entrada en los datos de validación y se mide cuánto cae el NSE. Muestra qué usa realmente el LSTM. No sirve para variables nuevas, y subestima las variables redundantes.
3. **Confirmación con el LSTM:**
   - configuraciones: el conjunto actual frente al conjunto propuesto, más ERA5-Land cuando esté;
   - 3 semillas y 3 épocas fijas cada una;
   - una noche de GPU.
4. **Prueba final:** una sola evaluación en 2016–2022 con el conjunto elegido.

**Antecedentes en la literatura:**
- Bakar et al. (2026) comparan el LSTM con y sin atributos estáticos.
- Kapoor y Chandra (2026) comparan con y sin variables de GR4J.
- Kratzert et al. (2021) comparan una lluvia frente a varias, con varias semillas.
- Kratzert et al. (2019) ordenan los 27 atributos de CAMELS con un análisis de sensibilidad del modelo entrenado.
- Heudorfer, Gupta y Loritz (2025) quitan los atributos estáticos del LSTM en CAMELS de EE. UU. Concluyen que la mayor parte del desempeño viene de las forzantes y que los atributos aportan poco, sobre todo en cuencas no vistas.

**Expectativa:** quitar variables rara vez sube mucho el NSE, porque el LSTM tolera entradas redundantes. Lo que puede mover el resultado es agregar información nueva, como la humedad del suelo. La ablación sirve sobre todo para sacar variables que identifican cuencas y para justificar el conjunto final en el informe.

**Acceso a los datos:**
- **Copernicus CDS:** cuenta creada el 17/11. La clave va en `C:\Users\Diego\.cdsapirc` y nunca en el chat ni en git.
- **Google Earth Engine:** sería la vía más eficiente para casi todo el catálogo, porque promedia por cuenca en el servidor y entrega tablas pequeñas. Tiene ERA5-Land diario, HydroATLAS, MapBiomas, Hansen, MODIS y AlphaEarth. Requiere otra cuenta gratuita y un proyecto de Google Cloud.
  - **Decisión (17/11): por ahora no hace falta.** Toda la prioridad A se consigue sin Earth Engine:
    - ERA5-Land, por el CDS;
    - El Niño, embalses y HydroATLAS, con descarga directa.
  - Solo sería necesario para AlphaEarth (fase auxiliar), y conveniente para MapBiomas anual, Hansen y MODIS (prioridad B y C).

## 17. ERA5-Land: de grillas a series diarias por cuenca (18/11)

**Qué:** se armó el paso de las grillas de ERA5-Land a una tabla diaria por cuenca y se corrió con los años ya descargados (1981–2009) mientras sigue la descarga.
- Código: desde el 19/11, en el notebook de preprocesamiento, secciones 2.2 (pesos) y 2.3 (series diarias). Antes era el script `procesar_era5land.py`.
- Tarda unos 5 minutos en total y usa poca RAM, porque lee cada archivo por bloques de un año.

**Cómo:**
- **Peso de cada celda:** el área de la celda que queda dentro de la cuenca (fracción × área de la celda sobre el elipsoide). Es el promedio ponderado de la sección 12.
- **Fracción:** se mide con 20 × 20 puntos por celda y la regla par-impar sobre todos los anillos del polígono, así que los huecos restan. Es un barrido por filas: da exactamente lo mismo que el método de la figura en 5 cuencas de prueba y calcula las 248 cuencas en 4 segundos.
- **Fecha:** día local (UTC−5).
  - Acumuladas (lluvia, radiación, evaporación, escorrentía): el total del día va de las 19:00 del día anterior a las 19:00 del día, hora local, porque ERA5-Land acumula por día UTC.
  - Humedad del suelo: el valor de las 07:00 hora local.
  - Rocío, viento y presión: el promedio de las muestras de las 01, 07, 13 y 19 h local.
- El viento (a partir de sus dos componentes) y la presión de vapor (a partir del rocío) se calculan en cada celda y cada hora antes de promediar, porque no son lineales.

**Salidas**, en `data/externos/era5land/`:
- `pesos_cuencas.csv`: celdas y pesos de cada cuenca. Son 23 067 pares cuenca-celda sobre 5569 celdas distintas; la mediana es de 16 celdas por cuenca, con un rango de 3 a 2261.
- `pesos_resumen.csv`: el área reconstruida de cada cuenca.
- `cuencas/*.npz`: el promedio por cuenca de cada archivo descargado, a la hora original.
- `era5land_cuencas_diario.parquet`: gauge_id, date y 15 variables, alineada fila por fila con `camels_col_diario.parquet` (3 804 320 filas). Por ahora está llena hasta 2009, el 69 % de las filas.
  - Lluvia, radiación solar y radiación térmica, evaporación real.
  - Escorrentía total, superficial y subsuperficial.
  - Humedad del suelo en 4 capas (0–7, 7–28, 28–100 y 100–289 cm).
  - Rocío, presión de vapor, viento y presión en superficie.

**Controles:**
- **Área:** polígono frente a atributo, mediana 0,992 (rango 0,948 a 1,008). Las más alejadas son cuencas pequeñas, por ejemplo 29067040, con 57 km² frente a 60. Ninguna cuenca toca celdas de mar.
- **Alineación de fechas:** la correlación diaria con la lluvia de CAMELS-COL (CHIRPS) es máxima sin desfase. La mediana es 0,325, frente a 0,24 y 0,19 al correr un día hacia atrás o hacia adelante.
- **Lluvia:** ERA5-Land da 1,5 veces la de CHIRPS en la mediana de cuencas (2934 frente a 1993 mm al año) y hasta 4 veces en el 5 % de cuencas con mayor exceso. La correlación de Spearman entre cuencas es apenas 0,30. Confirma el sesgo húmedo del piloto (sección 14).
  - La humedad del suelo y la escorrentía de ERA5-Land salen de esa misma lluvia. Pueden servir más por su dinámica (cuándo el suelo está húmedo o seco) que por sus montos.
- **Evaporación real de ERA5-Land frente a la ETP de CAMELS-COL:** mediana 0,96. Es alta, pero posible en cuencas húmedas.
- **Rocío:** la temperatura media de CAMELS-COL menos el rocío de ERA5-Land da 4,1 °C en la mediana, una humedad relativa de unos 75 %. Es coherente.
- **Vacíos:** ninguno dentro del período procesado, salvo el 31/12/2009 de rocío, viento y presión, que espera la muestra del 1/1/2010.

**Escorrentía de ERA5-Land:** la calcula el modelo de suelo de ERA5-Land con la lluvia y la meteorología del reanálisis, sin caudal observado. Cumple la regla de no usar caudal como entrada.

**Cuando termine la descarga:** correr `procesar_era5land.py espacial`, que solo procesa los archivos nuevos, luego `diario` y luego `revisar`.

## 18. Validación de los datos de ERA5-Land (18/11)

**Qué:** antes de usar ERA5-Land en la selección de variables, se comprobó que la tabla diaria por cuenca esté bien calculada.
- Se procesó hasta 2017 (descarga en 111 de 126 a las 17:30), así que entrenamiento (1981–2010) y validación (2011–2015) están completos.
- Código: desde el 19/11, en el notebook de preprocesamiento, sección 2.4 (antes, el script `validar_era5land.py`). Los valores recalculados quedan en `data/externos/era5land/validacion_recalculo.csv`.

**Resultados, 1981–2015:**

| Control | Resultado | Qué confirma |
|---|---|---|
| Completitud | 3 170 184 filas = 248 cuencas × 12 783 días; sin vacíos ni duplicados; mismo orden de filas que `camels_col_diario.parquet` | Que se puede unir fila por fila |
| Recálculo independiente desde las grillas originales: 9 cuencas, 54 pares cuenca-fecha, 810 valores. Usa otra forma de medir la fracción, peso por coseno de la latitud y horas UTC escogidas a mano. Incluye 1/1/1981, 31/12/1999, 29/2/2000, 31/12/2012 y 31/12/2015 | Diferencia máxima de 7 × 10⁻⁵ desviaciones estándar, del orden de la precisión de los decimales guardados | Pesos, horas, desfase de las acumuladas, promedio de las 4 muestras y unidades |
| Balance de agua del propio ERA5-Land: (lluvia − evaporación − escorrentía) / lluvia | Mediana −0,1 % (de −0,5 % a 0,0 %) | Signos y unidades de lluvia, evaporación y escorrentía |
| Escorrentía superficial + subsuperficial frente a total | Diferencia máxima de 2 × 10⁻⁶ mm/día | Que las tres escorrentías son coherentes |
| Presión media frente a la elevación media de la cuenca (atmósfera estándar de FAO-56) | r = 0,994; ERA5-Land 1 kPa más alta en la mediana | Que el promedio cae sobre la cuenca correcta |
| Radiación solar frente a la del tope de la atmósfera | Entre 0,12 y 0,74, mediana 0,48. Más baja en abril–mayo y octubre–noviembre, las dos temporadas de lluvia | Horas, unidades y ciclo anual |
| Rocío de ERA5-Land frente a la temperatura de CAMELS-COL | Rocío mayor que la máxima solo en 0,05 % de los días; humedad relativa mediana de 0,78 | Coherencia entre fuentes |
| Lluvia de ERA5-Land frente a la de ERA5 por Open-Meteo (30 cuencas del piloto, 2001–2010) | Correlación diaria mediana de 0,93 sin desfase, frente a 0,55 y 0,43 al correr un día; razón de medias 0,985 | Que es la misma lluvia del reanálisis y que las fechas están bien |
| Escorrentía de ERA5-Land frente al caudal observado (solo para evaluar, 1981–2010) | Correlación diaria mediana 0,30; NSE mediano −3,2; volumen 1,6 veces el observado | Que no sirve como simulación directa: si aporta, será por su dinámica |

**Conclusión:** la tabla está bien calculada. El problema conocido es el exceso de lluvia del reanálisis (sección 17), no el procesamiento.

**Detalle encontrado:** las coordenadas de las estaciones en `data/processed/atributos.csv` (`gauge_lat`, `gauge_lon`) están en metros (WGS 84 World Mercator), no en grados. Hay que convertirlas con la misma fórmula de los polígonos antes de usarlas como latitud.

## 19. Selección de variables con LightGBM (18/11)

**Diseño**, según el método de la sección 16:
- **Igual que T4 (sección 8.3):** las mismas variables derivadas, los mismos parámetros y los pesos tipo NSE*. Se entrena con 1981–2010 en días alternos (1 133 707 filas) y se evalúa en validación 2011–2015 (309 731 días de 195 cuencas). **La prueba 2016–2022 no se usa.**
- **800 árboles fijos, sin parada temprana.** T4 decidía cuándo parar mirando la validación (783 árboles). Aquí la validación es la que juzga las variables, así que no puede usarse también para parar. Se evalúa además con 400 árboles para ver si el orden de los experimentos depende del número de árboles.
- **Comprobación:** con las variables reconstruidas, el T4 guardado da NSE mediano 0,450 en validación. Las variables base coinciden con las de T4.
- **Ruido:** la línea base se entrena con 3 semillas. Cada experimento se compara, cuenca por cuenca, con el promedio de las 3.

**Experimentos (24):**
- Línea base con 3 semillas: 35 variables de forzantes y 48 atributos (83).
- ERA5-Land, todo junto y por grupo:
  - lluvia, con medias de 3 a 30 días e índices antecedentes;
  - humedad del suelo en 4 capas más el agua total en los 289 cm;
  - radiación solar y térmica;
  - humedad del aire: rocío, presión de vapor y déficit de presión de vapor con las temperaturas de CAMELS-COL;
  - viento y presión;
  - evaporación real;
  - escorrentía del modelo, con medias e índices antecedentes.
- Variables sin costo:
  - rango térmico diario y sus medias de 7 y 30 días;
  - latitud y longitud de la estación;
  - El Niño (ONI de NOAA, temporada de 3 meses que termina en el mes del día, sin mirar el futuro; `data/externos/oni/`);
  - estacionalidad de la lluvia de 1981–2010: índice de Walsh y Lawler, concentración y momento medio del año.
- Ablaciones (quitar y reentrenar):
  - evapotranspiración (`pet`, sus medias y el balance);
  - temperaturas;
  - `eroded_misce_perc`;
  - cada uno de los 6 grupos de atributos.

**Archivos:**
- Código: desde el 19/11, en el notebook de modelado, sección 8.4 (antes, el script `seleccion_lgbm.py`).
- Variables en bloques: `runs/seleccion_lgbm_1811/bloques/`, unos 700 MB, con `filas.parquet` y `columnas.json` (se movieron del scratchpad el 19/11).
- Resultados en `runs/seleccion_lgbm_1811/`:
  - `resultados.csv`, una fila por experimento;
  - `resultados_cuencas.csv`, el NSE y el KGE por cuenca;
  - `importancia.csv`;
  - `resumen.csv`, la comparación con la línea base.

**Cómo leer las diferencias.** Cada experimento se compara cuenca por cuenca con el promedio de las 3 semillas de la línea base. Se toma la mediana de esas diferencias y se aplica la prueba de Wilcoxon.
- La resta de medianas no sirve con una sola semilla: el NSE mediano de una semilla (0,448 a 0,452) queda por encima de la mediana del promedio por cuenca (0,446), así que inflaría cualquier experimento de una semilla en unas 0,004.
- **Ruido medido:** cada semilla de la línea base frente al promedio de las otras dos da una diferencia mediana por cuenca de −0,0001 a +0,0008, con 95 a 110 de 195 cuencas que mejoran y p de 0,52 a 0,94.
- **Umbral:** un efecto cuenta si la diferencia mediana supera unas 0,002, mejoran más de unas 115 cuencas y p < 0,01.

**Primera ronda, una semilla por experimento** (validación 2011–2015, 195 cuencas; línea base: NSE mediano del promedio de 3 semillas = 0,446):

| Experimento | Variables | Dif. mediana por cuenca | Cuencas que mejoran | p (Wilcoxon) | Con 400 árboles | Lectura |
|---|---|---|---|---|---|---|
| **+ todo ERA5-Land** | 117 | **+0,019** | **154** | < 10⁻⁴ | +0,017 | **Aporta claramente** |
| + radiación de ERA5-Land | 87 | +0,012 | 151 | < 10⁻⁴ | +0,010 | Aporta |
| + lluvia de ERA5-Land | 90 | +0,010 | 143 | < 10⁻⁴ | +0,008 | Aporta |
| + escorrentía de ERA5-Land | 91 | +0,009 | 141 | < 10⁻⁴ | +0,008 | Aporta |
| + evaporación de ERA5-Land | 87 | +0,004 | 128 | 10⁻⁴ | +0,004 | Aporta poco |
| + humedad del suelo de ERA5-Land | 88 | +0,004 | 130 | < 10⁻⁴ | +0,004 | Aporta poco |
| + viento y presión de ERA5-Land | 85 | +0,003 | 122 | < 10⁻⁴ | +0,003 | Aporta poco |
| + humedad del aire de ERA5-Land | 87 | +0,002 | 113 | 0,007 | +0,001 | En el límite |
| + rango térmico | 86 | +0,003 | 126 | 0,0002 | +0,002 | Aporta poco |
| + El Niño (ONI) | 84 | +0,002 | 105 | 0,014 | +0,002 | En el límite |
| + latitud y longitud | 85 | +0,001 | 103 | 0,45 | 0,000 | No aporta |
| + estacionalidad de la lluvia | 87 | +0,001 | 106 | 0,054 | +0,001 | No aporta |
| − evapotranspiración | 74 | **−0,004** | 78 | 0,006 | −0,004 | **Quitarla empeora: se queda** |
| − temperaturas | 77 | 0,000 | 98 | 0,39 | 0,000 | Redundante |
| − `eroded_misce_perc` | 82 | +0,001 | 104 | 0,14 | +0,001 | Quitarlo no empeora |
| − fisiográficos | 73 | −0,001 | 91 | 0,10 | −0,001 | Redundante |
| − climáticos | 76 | 0,000 | 94 | 0,27 | −0,002 | Redundante |
| − cobertura | 78 | −0,001 | 86 | 0,018 | −0,002 | En el límite |
| − geología | 77 | −0,001 | 91 | 0,12 | −0,001 | Redundante |
| − suelos | 70 | 0,000 | 98 | 0,42 | 0,000 | Redundante |
| − capacidad de uso | 76 | −0,001 | 90 | 0,50 | 0,000 | Redundante |

**Lectura de la primera ronda:**
- **ERA5-Land es la primera fuente nueva que mueve el resultado:** +0,019 de NSE mediano por cuenca, 154 de 195 cuencas mejoran, y lo sostiene igual con 400 árboles. En el piloto de la sección 14, solo la lluvia de ERA5 daba +0,003.
- La mejora viene de varios grupos a la vez (radiación, lluvia, escorrentía). Juntos dan más que cualquiera solo, así que cada uno aporta algo distinto.
  - La radiación probablemente funciona como indicador de nubosidad, es decir, de lluvia, independiente de CHIRPS.
- **Quitar variables casi nunca mejora,** como se esperaba (sección 16). Cada grupo de atributos es redundante con los demás. La evapotranspiración es la única cuya ausencia empeora con claridad.
- `eroded_misce_perc` se puede quitar sin perder nada, lo que refuerza la sospecha de que solo identifica cuencas.

**Segunda ronda: confirmación con 3 semillas y combinaciones** (cada fila es el promedio por cuenca de 3 semillas, frente al de la línea base):

| Experimento | Variables | NSE mediano | KGE mediano | Dif. mediana por cuenca | Cuencas que mejoran | Con 400 árboles |
|---|---|---|---|---|---|---|
| Línea base | 83 | 0,446 | 0,565 | — | — | — |
| + todo ERA5-Land | 117 | 0,480 | 0,590 | +0,020 | 155 | +0,016 |
| **+ todo ERA5-Land, − `eroded_misce_perc`** | **116** | **0,480** | **0,592** | **+0,020** | **158** | **+0,017** |
| + todo ERA5-Land + rango térmico + El Niño, − `eroded_misce_perc` | 120 | 0,479 | 0,586 | +0,019 | 158 | +0,018 |
| + suelo, radiación y lluvia de ERA5-Land | 99 | 0,479 | 0,587 | +0,016 | 153 | +0,015 |
| + radiación de ERA5-Land | 87 | 0,469 | 0,581 | +0,008 | 148 | +0,010 |
| + lluvia de ERA5-Land | 90 | 0,457 | 0,572 | +0,009 | 150 | +0,008 |
| + escorrentía de ERA5-Land | 91 | 0,455 | 0,572 | +0,008 | 150 | +0,007 |
| + evaporación de ERA5-Land | 87 | 0,460 | 0,570 | +0,005 | 133 | +0,004 |
| + humedad del suelo de ERA5-Land | 88 | 0,465 | 0,573 | +0,004 | 137 | +0,004 |
| + rango térmico | 86 | 0,451 | 0,566 | +0,002 | 120 | +0,002 |
| + El Niño (ONI) | 84 | 0,458 | 0,561 | +0,001 | 106 | +0,002 |

En todas las filas con ERA5-Land, p < 10⁻⁴ (Wilcoxon). Rango térmico: p = 0,0007; El Niño: p = 0,02, que no se confirma.

**Dónde ayuda ERA5-Land** (fila "+ rango térmico + El Niño, − `eroded_misce_perc`" frente a la base, 3 semillas cada uno; p = 4 × 10⁻¹⁹):
- **En todas las regiones:**
  - Pacífico: +0,052, de 0,47 a 0,56; mejoran 13 de 17 cuencas. Es donde CHIRPS se queda corto de lluvia.
  - Caribe: +0,021.
  - Magdalena-Cauca: +0,018, de 0,42 a 0,44.
  - Orinoco: +0,018.
  - Amazonas: +0,008.
- **En todos los grupos del diagnóstico** (`docs/diagnostico-cuencas.csv`). Lo más alto es "lluvia insuficiente", con +0,044 en 8 cuencas.
- **Cuencas con NSE negativo:** bajan de 10 a 7.
- **Ganancia de LightGBM:** las variables de ERA5-Land suman solo 4 % de la ganancia, aunque son las que mueven el NSE. Confirma que la ganancia no sirve para escoger variables (sección de registro de variables).

**Recomendación (por confirmar con el equipo y con el LSTM):**
- **LightGBM:** la línea base más todo ERA5-Land, sin `eroded_misce_perc` (116 variables). Es la combinación más simple que conserva toda la mejora.
- **El Niño, el rango térmico, latitud y longitud, y la estacionalidad de la lluvia:** no entran. Lo que aportan solos lo cubre ERA5-Land, o no aportan.
- **Se quedan:** la evapotranspiración, porque quitarla empeora, y los 6 grupos de atributos. Quitar un grupo no empeora, pero tampoco mejora, y los atributos se probarán aparte en el flujo espacial.
- **LSTM:** agregar las 15 variables diarias de ERA5-Land como forzantes y quitar `eroded_misce_perc`. El LSTM arma por sí mismo las medias y los índices antecedentes que LightGBM necesita como columnas.
- **Pendiente (paso 3 del método):** confirmar con el LSTM, 3 semillas por configuración. Si la mejora se sostiene, recalcular el promedio LSTM + LightGBM y evaluar la prueba una sola vez.

**Atributos en cuencas no vistas: falta decidir cómo probarlos.**
- Las 95 cuencas que se dejan fuera en las 5 particiones espaciales son las de prueba del flujo espacial. Si se usan para escoger atributos, esa prueba queda contaminada.
- **Opción limpia:** particiones internas solo con las 153 cuencas fijas. Como están casi todas anidadas en dos grandes sistemas, hay que agruparlas por anidamiento.
- Decisión del equipo.

## 20. Importancia por permutación en el LSTM (18/11)

**Qué:** paso 2 del método de la sección 16. Se toma el LSTM ya entrenado (T1, época 3), se revuelve una entrada o un grupo y se mide cuánto cae el NSE en validación 2011–2015 (195 cuencas). No se reentrena.
- **Forzantes:** se permutan los días de 2010–2015 dentro de cada cuenca (la ventana de 365 días más la validación), solo entre días con dato, así que los huecos quedan donde estaban.
- **Atributos:** se permutan entre cuencas, con la misma permutación para todas las columnas del grupo.
- Se usa el evaluador de NeuralHydrology sin cambios; solo se intercepta la lectura de datos.
- Código: desde el 19/11, en el notebook de modelado, sección 8.5 (antes, el script `permutacion_lstm.py`). Resultados en `runs/seleccion_lgbm_1811/permutacion_lstm.csv`. Unos 16 minutos en total.

**Resultados** (NSE mediano sin revolver: 0,460, igual que en el entrenamiento):

| Se revuelve | NSE mediano | Caída mediana por cuenca | Cuencas que empeoran más de 0,005 | En LightGBM, quitarlo (sección 19) |
|---|---|---|---|---|
| `pr` | −0,107 | −0,531 | 189 | (no se probó: es la entrada principal) |
| Los 48 atributos | 0,122 | −0,326 | 178 | — |
| Fisiográficos (10) | 0,332 | −0,053 | 135 | −0,001 |
| Climáticos (7) | 0,365 | −0,051 | 143 | 0,000 |
| Suelos (13) | 0,382 | −0,038 | 125 | 0,000 |
| Capacidad de uso (7) | 0,393 | −0,029 | 126 | −0,001 |
| Geología (6) | 0,407 | −0,028 | 130 | −0,001 |
| `poten_evapo` | 0,413 | −0,024 | 135 | −0,004 |
| Cobertura (5) | 0,414 | −0,023 | 132 | −0,001 |
| `t_min` y `t_max` | 0,440 | −0,007 | 103 | 0,000 |
| `eroded_misce_perc` | 0,460 | 0,000 | 14 | +0,001 |

**Lectura:**
- **El LSTM se apoya mucho más en los atributos que LightGBM.** Revolver todos los atributos baja el NSE de 0,46 a 0,12. Cada grupo resta de 0,02 a 0,05 por cuenca. En LightGBM, quitar un grupo y reentrenar no cambiaba nada.
  - Parte de la diferencia es del método. Al revolver se crean cuencas imposibles (por ejemplo, la elevación de una y la aridez de otra), y el modelo no puede compensar con otras variables como cuando se reentrena. Por eso la permutación exagera la importancia de variables correlacionadas.
  - Aun así, muestra que el LSTM sí usa los atributos para distinguir cuencas. Eso es lo que se espera de un modelo regional, pero también es el riesgo de que solo identifique cuencas en el flujo temporal (Heudorfer et al., 2025). **Los atributos no se pueden quitar sin probar en cuencas no vistas** (sección 19, punto pendiente).
- **Evapotranspiración frente a temperaturas:** el LSTM usa `poten_evapo` (−0,024) bastante más que `t_min` y `t_max` (−0,007), aunque `poten_evapo` sale de esas mismas temperaturas. Le sirve la combinación ya hecha, con la radiación teórica del día del año. Coincide con LightGBM: quitar la evapotranspiración empeora y quitar las temperaturas no.
- **`eroded_misce_perc` no la usa el LSTM** (caída nula; solo 14 cuencas cambian). Junto con el resultado de LightGBM, **se puede quitar en los dos modelos.**

## 21. Base para el LSTM con ERA5-Land (18/11)

**Qué:** copia de `data/processed` con las 15 variables diarias de ERA5-Land agregadas a la serie de cada cuenca, para la confirmación con el LSTM (paso 3 del método). No toca `data/processed`.
- Carpeta: `data/processed_era5land/`, con `time_series/` (248 archivos) y `attributes/`.
- Código: desde el 19/11, en el notebook de preprocesamiento, sección 2.5 (antes, el script `base_lstm_era5land.py`).
- ~~ERA5-Land está completo hasta el 30/12/2017.~~ **Reconstruida el 19/11 (07:05) con ERA5-Land completo de 1981 a 2022**, sin huecos. Alcanza para la prueba 2016–2022.
  - **Control:** se reevaluó la validación de `confirmacion_era5l_s42` (época 3) con la base nueva, sin guardar nada. El NSE de las 195 cuencas es idéntico (diferencia máxima 4e-15), así que la reconstrucción no cambió los datos de 1981–2017 con que se entrenó.

**Confirmación con el LSTM (lanzada el 18/11 a las 22:32):**
- **Dos configuraciones**, todo lo demás igual a T1 (lote 256, 256 celdas, tasa 0,0005, entrenamiento 1981–2010):
  - `base`: la actual, 4 forzantes y 48 atributos, leyendo `data/processed`;
  - `era5l`: 4 forzantes, más las 15 de ERA5-Land, y 47 atributos, sin `eroded_misce_perc`, leyendo `data/processed_era5land`.
- **Cómo se entrena:** semillas 42, 43 y 44; 3 épocas fijas, sin parada temprana y con tasa de aprendizaje constante.
  - Las corridas se alternan (`era5l` y `base` de cada semilla), para que, si la noche se corta, queden pares completos.
  - Si una corrida falla, se reintenta una vez.
- **Cómo se compara**, en validación 2011–2015 (195 cuencas):
  - en la época 3, fijada de antemano; las épocas 1 y 2 se guardan solo como referencia;
  - con la diferencia mediana por cuenca del promedio de las 3 semillas, cuántas cuencas mejoran y Wilcoxon;
  - también con el ensamble de las 3 semillas (promedio de las simulaciones), por semilla y por región;
  - el ruido se mide con pares de la misma configuración y distinta semilla;
  - control de reproducibilidad: `base` con semilla 42 debe dar casi lo mismo que T1 en la época 3 (0,460).
- **Antes de lanzar:**
  - una prueba de humo de `era5l` (25 cuencas, 1 época) corrió sin errores;
  - el NSE recalculado desde las simulaciones guardadas coincide con el de NeuralHydrology (diferencia de 1e-16).
- **Archivos:**
  - configuraciones en `configs/corridas/confirmacion_{base,era5l}_s{42,43,44}.yml`;
  - corridas en `runs/confirmacion_*`;
  - avance, resumen y NSE por cuenca en `runs/confirmacion_lstm_1811/` (`progreso.log`, `resumen.csv`, `cuencas.csv`);
  - código: desde el 19/11, en el notebook de modelado, sección 8.6 (antes, el script `confirmacion_lstm.py`).

**Resultado (19/11; terminó a las 03:55, sin fallos).** Cada corrida con ERA5-Land tardó 58–59 minutos y cada una sin ERA5-Land, 49.
- **Control:** `base` con semilla 42 reproduce T1 exacto (NSE 0,460; diferencia por cuenca 0,0000). El entrenamiento es determinista.

Validación 2011–2015, 195 cuencas, época 3. La diferencia es `era5l` − `base`, por cuenca.

| Comparación | NSE `base` | NSE `era5l` | Diferencia mediana | Cuencas que mejoran | p (Wilcoxon) |
|---|---|---|---|---|---|
| **Promedio de las 3 semillas por cuenca** | 0,452 | **0,491** | **+0,024** | **145 de 195** | 2e-12 |
| Ensamble (promedio de las 3 simulaciones) | 0,468 | **0,505** | +0,024 | 146 | 9e-13 |
| Semilla 42 | 0,460 | 0,482 | +0,027 | 127 | 2e-5 |
| Semilla 43 | 0,445 | 0,489 | +0,023 | 132 | 4e-8 |
| Semilla 44 | 0,446 | 0,509 | +0,028 | 131 | 1e-8 |
| Ruido: misma configuración, otra semilla (6 pares) | — | — | −0,012 a 0,000 | 82 a 97 | 0,006 a 0,93 |

- **La mejora es clara:** la sostienen las 3 semillas, y es unas 2 veces el mayor cambio entre semillas y unas 10 veces el cambio típico. También aparece en las épocas 1 y 2 (+0,033 y +0,026 con el promedio de semillas).
- **Por región** (promedio de semillas):

  | Región | Cuencas | Diferencia mediana |
  |---|---|---|
  | Amazonas | 9 | +0,038 |
  | Caribe | 25 | +0,023 |
  | Magdalena-Cauca | 113 | +0,023 |
  | Orinoco | 31 | +0,023 |
  | Pacífico | 17 | +0,050 |

  - En el Pacífico pesa mucho la semilla: −0,057 con la 42, +0,087 con la 43 y +0,078 con la 44.
- **Por cuenca:**
  - cuartiles de la diferencia: −0,001, +0,024 y +0,065;
  - 68 cuencas ganan más de 0,05 y 12 pierden más de 0,05;
  - cuencas con NSE ≥ 0,5: de 80 a 94; con NSE ≥ 0,7: de 32 a 33;
  - cuencas con NSE < 0: 13 y 14, sin cambio.
- **Coincide con LightGBM** (sección 19: +0,020 por cuenca, 158 de 195). Los dos modelos aprovechan ERA5-Land en la misma medida.
- **La mejora es pareja en todos los niveles de desempeño.** Por tramo del NSE de `base`:

  | NSE de `base` | Cuencas | Diferencia mediana | Mejoran |
  |---|---|---|---|
  | < 0 | 13 | +0,063 | 9 |
  | 0 a 0,3 | 42 | +0,025 | 29 |
  | 0,3 a 0,5 | 60 | +0,023 | 48 |
  | 0,5 a 0,7 | 48 | +0,028 | 36 |
  | ≥ 0,7 | 32 | +0,019 | 23 |

- **Las 13 cuencas con NSE negativo suben, pero 11 siguen por debajo de 0.** Solo `13067020` pasa a positivo (de −0,16 a 0,08). `24017820` (−16,7) y `16037040` (−4,0) siguen muy mal.
  - En el diagnóstico de la prueba (`docs/diagnostico-cuencas.csv`) hay 10 de ellas: 6 quedan en "Resto", 2 en "relación lluvia-caudal cambió", 1 en "lluvia insuficiente" y 1 en "mala sincronía".
  - Su problema no queda explicado por los grupos del diagnóstico; falta revisarlas una por una.
- **Recomendación:** que el LSTM use la configuración `era5l`, con 19 forzantes y 47 atributos.

- **Después:** como la mejora se sostiene, se entrena el LightGBM recomendado (sección 19), se recalcula el promedio LSTM + LightGBM y se evalúa la prueba una sola vez (sección 22).

---

## 22. Evaluación final en la prueba 2016–2022 (19/11)

**Qué:** la única evaluación en la prueba del modelo elegido en validación (secciones 19 y 21). Se aprobó el 19/11.

**Modelo final, fijado antes de mirar la prueba:** el promedio de dos ensambles.
- **LSTM:** el ensamble de las 3 corridas `era5l` (C2; semillas 42, 43 y 44; época 3), es decir, el promedio de sus 3 simulaciones. No se reentrena.
- **LightGBM:** el ensamble de 3 modelos con la configuración recomendada (sección 19): base + ERA5-Land completo − `eroded_misce_perc` (116 variables), con 800 árboles fijos, las semillas 42, 43 y 44 y entrenamiento con 1981–2010 en días alternos.
- **Final:** el promedio de los dos ensambles, en los días que el LSTM puede simular.

**Cómo se evalúa** (igual que T4 y T5, sección 8.3 del notebook):
- 183 cuencas de `prueba.txt`, días con caudal observado de 2016–2022, al menos 365 días evaluables por cuenca;
- métricas de NeuralHydrology con la simulación recortada en cero;
- medianas entre cuencas.

**Referencias**, reportadas junto al final pero sin usarlas para escoger:
- T1, T4 y T5, recalculadas para comprobar que el cálculo reproduce 0,394, 0,390 y 0,410;
- C1, el LSTM con las variables actuales en ensamble de 3 semillas, solo y con LightGBM base (3 semillas, 800 árboles), para medir cuánto aporta ERA5-Land en la prueba.

**Después de esta evaluación no se cambia nada del modelo mirando la prueba.**

**Cómo corrió (19/11, de 07:20 a 08:17):**
- **LSTM:** la prueba de las 6 corridas de la confirmación se evaluó con NeuralHydrology, entre 1,7 y 2,6 minutos por corrida.
- **LightGBM:** se entrenaron 6 modelos, de 1,5 a 2 minutos cada uno. El primero tardó 32 minutos, porque leyó por primera vez los bloques del disco.
  - Cada modelo reproduce al decimal el NSE de validación de la selección; por ejemplo, 0,4510 en la configuración actual con semilla 42 y 0,4810 con ERA5-Land y la misma semilla.
- **Controles:**
  - T1, T4 y T5 se reproducen exactos: 0,394, 0,390 y 0,410;
  - T1 y C2 tienen simulación en exactamente los mismos días, así que todos los modelos se comparan sobre los mismos días.
- **Archivos en `runs/prueba_final_1911/`:**
  - `resumen.csv`, `comparaciones.csv` y `metricas_cuencas.csv`;
  - `series_final.parquet`: observado y simulado del modelo final, día a día;
  - `lgbm_*.txt`: los 6 modelos;
  - `lgbm_prueba.parquet`: las predicciones de LightGBM.
- **Código:** desde el 19/11, en el notebook de modelado, sección 8.7 (antes, el script `prueba_final.py`).

**Resultado.** Prueba 2016–2022, 183 cuencas, mismos días, medianas entre cuencas.

| Modelo | NSE | KGE | NSE < 0 | NSE ≥ 0,5 | NSE ≥ 0,7 |
|---|---|---|---|---|---|
| T1: LSTM, 1 semilla | 0,394 | 0,513 | 20 | 56 | 21 |
| T4: LightGBM | 0,390 | 0,525 | 20 | 52 | 21 |
| T5: T1 + T4 (mejor hasta el 17/11) | 0,410 | 0,518 | 20 | 59 | 21 |
| C1: LSTM actual, ensamble de 3 semillas | 0,389 | 0,511 | 20 | 58 | 18 |
| LightGBM actual, 3 semillas | 0,396 | 0,525 | 20 | 54 | 21 |
| C1 + LightGBM actual | 0,415 | 0,534 | 19 | 63 | 19 |
| C2: LSTM con ERA5-Land, ensamble de 3 semillas | 0,432 | **0,539** | **14** | 72 | **23** |
| LightGBM con ERA5-Land, 3 semillas | 0,417 | 0,529 | 18 | 64 | 21 |
| **FINAL: C2 + LightGBM con ERA5-Land** | **0,437** | 0,536 | 17 | **72** | 22 |
| LightGBM con ERA5-Land, todos los días con caudal | 0,412 | 0,518 | 16 | 59 | 20 |
| Meta de la propuesta | 0,70 | 0,70 | — | — | — |

Diferencias por cuenca (A − B):

| Comparación | Métrica | Diferencia mediana | Cuencas que mejoran (de 183) | p (Wilcoxon) |
|---|---|---|---|---|
| **Final − T5** | NSE | **+0,020** | **127** | 1e-8 |
| Final − T5 | KGE | +0,002 | 95 | 0,38 |
| Final − (C1 + LightGBM actual): efecto de ERA5-Land en el promedio | NSE | +0,020 | 133 | 8e-11 |
| C2 − C1: efecto de ERA5-Land en el LSTM | NSE | +0,024 | 120 | 1e-7 |
| C2 − C1 | KGE | +0,009 | 110 | 0,02 |
| LightGBM con ERA5-Land − LightGBM actual | NSE | +0,015 | 126 | 3e-10 |
| C1 − T1: ensamble de 3 semillas frente a 1 | NSE | +0,008 | 110 | 0,01 |
| Final − C2: qué agrega LightGBM al LSTM con ERA5-Land | NSE | +0,002 | 103 | 0,04 |
| Final − C2 | KGE | −0,005 | 76 | 0,05 |

**NSE mediano por región:**

| Región | T5 | Final |
|---|---|---|
| Amazonas | 0,64 | 0,70 |
| Orinoco | 0,69 | 0,71 |
| Pacífico | 0,43 | 0,50 |
| Caribe | 0,35 | 0,38 |
| Magdalena-Cauca | 0,36 | 0,38 |

**Lectura:**
- **ERA5-Land se sostiene en la prueba**, con el mismo tamaño que en validación: +0,024 en el LSTM (validación: +0,024) y +0,015 en LightGBM (validación: +0,020). La selección en validación no se sobreajustó.
- **El modelo final mejora frente al mejor anterior (T5):** NSE de 0,410 a 0,437; 127 de 183 cuencas mejoran; cuencas con NSE ≥ 0,5 de 59 a 72; con NSE < 0 de 20 a 17.
  - Cuartiles de la diferencia por cuenca: −0,005, +0,020 y +0,052. 50 cuencas ganan más de 0,05 y 14 pierden más de 0,05.
- **El KGE casi no se mueve** (0,518 a 0,536; por cuenca, +0,002, p = 0,38). Falta ver cuál de sus componentes lo frena: correlación, variabilidad o volumen. En las cuencas con deriva el volumen pesa mucho (sección 10).
- **Con ERA5-Land, LightGBM casi no agrega al LSTM:** +0,002 de NSE y −0,005 de KGE. Antes el promedio era la única mejora significativa (T5 frente a T1, +0,016). Ahora el LSTM con ERA5-Land solo (0,432, con el mejor KGE y la menor cantidad de cuencas con NSE < 0) es casi igual de bueno y más simple.
- **Por región:** Amazonas y Orinoco quedan en 0,70 a 0,71 y el Pacífico sube a 0,50. Magdalena-Cauca (108 cuencas) y Caribe siguen cerca de 0,38: ahí está el techo.
- **Sigue lejos de la meta de 0,70.** Las causas de las secciones 10 y 13 (deriva en la relación lluvia-caudal, balance inconsistente) no se resuelven con más forzantes.

---

## 23. Alternativas para mejorar el modelo; transfer learning y fine-tuning (19/11)

**Qué:** con ERA5-Land el modelo llegó a 0,437 (sección 22). Se catalogaron las alternativas que quedan, de la que menos ganancia promete a la que más, con su evidencia, costo y riesgos, y se exploró en detalle el transfer learning y el fine-tuning. **El catálogo completo, con fuentes, está en `docs/alternativas-mejora.csv`.**

**Criterio del orden:** la ganancia esperada en el NSE mediano de prueba del flujo temporal, según la literatura y nuestras propias pruebas. Cuando el beneficio esperado está sobre todo en el flujo espacial (cuencas no aforadas), se dice aparte. Las ganancias son estimaciones, no mediciones.

**Advertencia:** ninguna alternativa, sola ni sumada, se espera que llegue a 0,70. El techo está en Magdalena-Cauca y el Caribe, por la deriva de la relación lluvia-caudal y el balance inconsistente (secciones 10 y 13), y eso no lo arreglan más forzantes ni otra red.

| Orden | Alternativa | Ganancia esperada (temporal) | Costo | Evidencia |
|---|---|---|---|---|
| 1 | **Otra arquitectura:** Transformer, Mamba o xLSTM, que ya trae NeuralHydrology 1.13 | ≈ 0 | Bajo: cambiar la configuración; 1 noche por arquitectura | En CAMELS el LSTM sigue siendo el mejor para caudal diario; los Transformers ganan en pronósticos a varios días (Liu et al., 2025) |
| 2 | **Manejo de huecos de forzantes** (`nan_handling_method`) | ≈ 0 en NSE; simula el 100 % de los días | Bajo: 1 corrida | Gauch et al. (2025): `masked_mean` es levemente mejor y es lo que usa Google. Nuestro relleno (T6) amplió la cobertura sin mejorar la precisión |
| 3 | **Más semillas** en el ensamble (5 a 10) | +0,002 a +0,005 | 1 hora de GPU por semilla | De 1 a 3 semillas se ganó +0,008 en la prueba (sección 22); el beneficio decrece |
| 4 | **Años recientes:** entrenar con 1991–2015 | 0 a +0,01 | 1 a 2 horas | T3 sumó +0,004, no significativo; Yu y Tolson (2026) |
| 5 | **Ajuste de hiperparámetros:** tamaño, ventana, dropout, tasa decreciente | 0 a +0,01 | 1 noche por 4 a 6 configuraciones | La mejor época siempre es la 3, señal de una tasa alta; el lote no cambió nada |
| 6 | **Fine-tuning por región** (Magdalena-Cauca, Caribe) | 0 a +0,01 en esas regiones | Menos de 1 hora por región | He et al. (2024): un modelo regional único supera a modelos por clase; el fine-tuning conserva lo aprendido con todas |
| 7 | **Otra lluvia independiente** (MSWEP v2.8, PERSIANN-CDR) | +0,005 a +0,02 | Medio: licencia académica de MSWEP; el procesamiento del preprocesamiento (2.2 y 2.3) sirve igual | La lluvia de ERA5-Land sumó +0,009 en LightGBM; varias lluvias se complementan (Kratzert et al., 2021) |
| 8 | **Intervención humana:** capacidad de embalses aguas arriba según su año de construcción (Global Dam Watch) y cobertura anual (MapBiomas) | +0,005 a +0,02; más en las 48 cuencas con deriva | Medio | Ouyang et al. (2021): el grado de regulación cambia mucho el desempeño del LSTM |
| 9 | **Fine-tuning por cuenca** del modelo regional | +0,01 a +0,03 (incierto) | 8 a 15 horas para todas | Ryd y Nearing (2025): ajustar un modelo global a cada cuenca mejora el desempeño |
| 10 | **Transfer learning desde Caravan** | 0 a +0,03 en el temporal; probablemente la mayor ganancia en cuencas no aforadas | Alto: 2 a 3 noches de GPU, más la preparación de datos | Ma et al. (2021); Benín (2026); Alzhanov et al. (2025, beneficio limitado) |
| — | Embeddings de AlphaEarth (notebook, sección 12) | ≈ 0; algo en el espacial | Bajo-medio | Qu et al. (2026) |
| — | Modelo híbrido diferenciable: un LSTM que estima los parámetros de un modelo conceptual (`hybridmodel` en NeuralHydrology) | ≈ 0 en NSE; más coherencia física | Medio-alto | Feng et al. (2022); Acuña Espinoza et al. (2024) |

**Descartadas por la regla del proyecto** (no usar caudal observado ni nada derivado de él como entrada):
- GR4J calibrado con nuestro caudal;
- el caudal de cuencas donantes;
- la simulación del modelo global de Google (GRRR, Flood Hub). Ese modelo se entrenó con caudal observado de GRDC, que desde 2024 tiene datos de 34 estaciones del IDEAM; podría contener nuestro propio caudal.

### Transfer learning y fine-tuning

**Qué es cada uno:**
- **Fine-tuning:** partir de un modelo ya entrenado y seguir entrenándolo, con una tasa de aprendizaje baja y pocas épocas, solo con los datos de un dominio más chico: una región o una cuenca.
  - Puede reentrenar todo el modelo o solo algunas partes. En el CudaLSTM de NeuralHydrology las partes son la capa de entrada (`embedding_net`), el `lstm` y la cabeza (`head`).
  - NeuralHydrology 1.13 lo trae: modo `finetune`, con `base_run_dir` y `finetune_modules`. Usa el escalado del modelo base.
- **Transfer learning:** preentrenar con un conjunto de datos más grande y de otra procedencia (Caravan), y después ajustar con el nuestro. También se puede entrenar con los dos juntos.

**Qué dice la evidencia:**
- **Ma et al. (2021):** preentrenaron con Estados Unidos y ajustaron en Gran Bretaña, Chile (49 cuencas) y China. Mejora con claridad donde hay pocos datos, más que preentrenar con las salidas de un modelo hidrológico. Bastan 50 a 100 cuencas de origen.
- **Benín (EGUsphere, 2026):** escogieron cuencas donantes de Caravan por similitud climática (tropicales). El subconjunto más chico y más parecido (50 cuencas) dio el mejor ajuste; agregar cuencas menos parecidas no ayudó. El beneficio fue mayor en estaciones con registros cortos.
- **Ryd y Nearing (2025):** ajustar un modelo global de Caravan a cada cuenca, en 159 cuencas, mejora el desempeño.
- **Kratzert et al. (2024):** entrenar con muchas cuencas siempre gana a entrenar con una. Recomiendan entrenar con todo y luego ajustar cada cuenca, aunque no lo prueban.
- **Asia Central (tesis de maestría, 2025):** el beneficio depende de la arquitectura y del tipo de cuenca.
- **Alzhanov et al. (2025, en la propuesta):** beneficio limitado.
- **Lectura para nosotros:** el transfer learning ayuda sobre todo cuando faltan datos. CAMELS-COL ya tiene 248 cuencas con 30 años, así que en el flujo temporal se espera poco. **Donde más puede servir es en el flujo espacial:** un modelo que vio más cuencas tropicales distintas debería generalizar mejor a cuencas sin estación.

**Qué haría falta para preentrenar con Caravan:**

| Requisito | Situación | Qué haría falta |
|---|---|---|
| Mismas forzantes, calculadas igual | Caravan agrega ERA5-Land al día local, como nuestro preprocesamiento (sección 2). Coinciden la lluvia, el rocío, la presión, la humedad del suelo en 4 capas y el viento (desde sus dos componentes) | Caravan trae radiación **neta** y temperatura de ERA5-Land; nosotros bajamos radiación **descendente** y usamos las temperaturas de CAMELS-COL (MSWX). Hay dos caminos: bajar temperatura a 2 m y radiación neta solar y térmica (una descarga más, de alrededor de 1 día) o preentrenar solo con las variables comunes |
| Evapotranspiración | La de Caravan es irrealmente alta en muchas cuencas (Clerc-Schwarzenbach et al., 2024) | Calcularla igual en los dos lados, con Hargreaves o FAO-56 |
| Mismos atributos | Caravan usa HydroATLAS y unos índices climáticos de ERA5-Land; nosotros, los 47 de CAMELS-COL | Un conjunto común: índices climáticos calculados con las forzantes (iguales en los dos lados), área y elevación. Alternativa: calcular HydroATLAS para nuestras cuencas con los polígonos de BasinATLAS |
| Cuencas donantes | Caravan v1.6 tiene unas 22 000 cuencas con sus extensiones (24,8 GB). En Sudamérica son sobre todo de Chile, Brasil y Argentina (GRDC-Caravan) | Escoger cuencas de clima tropical o parecido. Según Benín, 50 a 150 parecidas sirven más que miles distintas |
| Sin fuga de información | GRDC recibió en 2024 el caudal de 34 estaciones del IDEAM. GRDC-Caravan (2025) podría incluir estaciones colombianas, algunas quizás nuestras | Excluir toda cuenca de Caravan dentro de Colombia o que se cruce con las nuestras |
| RAM | NeuralHydrology carga todo en memoria. Con 7,8 GB caben unas 1000 a 1500 cuencas de 30 años | Usar un subconjunto de cuencas o Google Colab |
| Pesos preentrenados | No se encontraron pesos públicos de un LSTM de Caravan | Preentrenar nosotros. Ryd y Nearing tardaron unas 30 horas en una GPU con 6375 cuencas; con unas 500 cuencas, sería una noche |

**Plan propuesto, por fases** (cada fase decide si vale la siguiente):
1. **Fase 0, inventario sin GPU (1 a 2 horas).**
   - Bajar solo las tablas de atributos de Caravan y GRDC-Caravan, que pesan pocos MB.
   - Contar las cuencas tropicales de Sudamérica, revisar si hay estaciones colombianas, y ver qué variables y años trae.
2. **Fase 1, fine-tuning dentro de CAMELS-COL (1 noche),** partiendo de los 3 modelos C2:
   - **por región:** Magdalena-Cauca y Caribe, reentrenando el `lstm` y la `head` con tasa 1 × 10⁻⁴ durante 1 a 2 épocas;
   - **por cuenca:** una muestra de unas 30 cuencas de validación, estratificada por grupo del diagnóstico.

   Se mide en validación 2011–2015 frente a C2; la prueba no se toca.
3. **Fase 2, preentrenamiento con Caravan (2 a 3 noches, más la preparación de datos).** Tres variantes, todas con las mismas entradas comunes:
   - **(a)** solo CAMELS-COL, como referencia, porque cambian las entradas;
   - **(b)** Caravan y CAMELS-COL juntos;
   - **(c)** preentrenar con Caravan y ajustar con CAMELS-COL: todo el modelo, o solo la `head` y la `embedding_net`.

   Se mide en la validación del flujo temporal y en cuencas no vistas.
4. **Antes de la fase 2 hace falta el flujo espacial base** (sección 9 del notebook, con la configuración C2, unas 20 horas): sin él no hay con qué comparar en cuencas no vistas.

**Recomendación:** hacer las fases 0 y 1, que son baratas y dicen si el fine-tuning sirve en nuestros datos. La fase 2 solo vale la pena si el equipo quiere apostar por el flujo espacial, que es donde el transfer learning tiene más evidencia. En el temporal, el fine-tuning por cuenca (fase 1) es la apuesta más barata con evidencia a favor.

**Fuentes:**
- Ma et al. (2021), WRR: https://par.nsf.gov/servlets/purl/10270718
- Ryd y Nearing (2025), ICLR, taller CCAI: https://arxiv.org/abs/2504.12559
- Benín, Ouémé (EGUsphere, 2026): https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4749/
- Asia Central (tesis de maestría, 2025): https://static.gruppocdt.ch/utils-uploads/2026/03/01/9b80b3b8-cfb2-49bf-b802-931fc79c6a65/MasterThesisLazaro2025HSOL-2.pdf
- Kratzert et al. (2024), HESS: https://hess.copernicus.org/articles/28/4187/2024/
- Gauch et al. (2025), HESS: https://research.google/pubs/how-to-deal-w___-missing-input-data/
- Caravan v1.6: https://zenodo.org/records/15529786
- GRDC-Caravan (ESSD, 2025): https://essd.copernicus.org/articles/17/4613/2025/
- Caravan MultiMet: https://arxiv.org/abs/2411.09459
- Evapotranspiración de Caravan (HESS, 2024): https://hess.copernicus.org/articles/28/4219/2024/
- GRDC, datos de Colombia (2024): https://grdc.bafg.de/news/2024/new_data_co/index.html
- LSTM frente a Transformers en cuencas no aforadas: https://arxiv.org/pdf/2606.02791
- Ouyang et al. (2021), embalses: https://arxiv.org/abs/2101.04423
- Google, modelos de Flood Hub: https://sites.research.google/gr/floodforecasting/our-models/

## 24. Etapa 2: cuencas no vistas (27 y 28/11; notebook, secciones 7, 8 y 10)

**Qué se hizo.** Se entrenó por primera vez el flujo espacial, que estaba implementado desde el 11/11 pero nunca se había corrido. El plan está en `docs/plan-accion-cuencas-sin-estacion.md`.

- **Diseño, fijado antes de ver resultados:**
  - las 5 particiones de 19 cuencas por sistema de ríos (sección 5);
  - el protocolo C1 de la sección 21: 3 épocas fijas, sin parada temprana;
  - semillas 42, 43 y 44 en ensamble;
  - la referencia de cuenca vista es la corrida C1 (o C2) de cada semilla, en las mismas 62 cuencas y días de 2016–2022.
- **Código nuevo en `src/`:**
  - `flujo_espacial.py`: entrena y evalúa. Variantes `base`, `era5l` y `transformer`; salta las corridas ya evaluadas.
  - `baselines_espacial.py`: tres referencias por partición.
  - `resultados_espacial.py`: métricas en los mismos días, Wilcoxon, por región y por partición.
  - `figuras_espacial.py`: CDF, mapa e hidrogramas.
- **Tres referencias**, cada una con solo las cuencas de entrenamiento de su partición:
  - climatología de 5 donantes por similitud de atributos;
  - vecino más cercano con caudal observado;
  - LightGBM regional con las 83 variables de la sección 8.3.
- **Error corregido:** `gauge_lat` y `gauge_lon` están en metros de World Mercator (EPSG:3395), no en grados. Se convierten con `mercator_a_grados`. El vecino queda a 44 km en mediana; leídas como grados, saldría a 653 km.
- **Datos:** la base de ERA5-Land (`data/processed_era5land`) la pasó Diego el 27/11 y se validó: 248 cuencas, 1981–2022.
- **Tiempo de entrenamiento** (RTX 3050): 33 min por corrida en la variante base y 42 min en la de ERA5-Land. El Transformer va a unos 85 min por corrida.
- **Incidente:** el 27/11 a las 20:05, el temporizador `apagado-nocturno` apagó el equipo a mitad de una corrida de ERA5-Land. Se reanudó a las 22:38 y el script saltó las corridas ya hechas.

**Resultados en las 95 cuencas no vistas, 1981–2022** (medianas; NSE = R² por cuenca):

| Modelo | NSE | KGE | Cuencas con NSE < 0 | Cuencas con NSE ≥ 0,5 |
|---|---|---|---|---|
| LSTM base, ensamble de 3 semillas | 0,276 | 0,438 | 22 | 24 |
| **LSTM con ERA5-Land, ensamble de 3 semillas** | **0,300** | **0,453** | 21 | 26 |
| LightGBM regional | 0,257 | 0,445 | 25 | 24 |
| Promedio LSTM base + LightGBM | 0,276 | 0,436 | 19 | 27 |
| Vecino más cercano (caudal observado) | −0,426 | 0,231 | 65 | 14 |
| IDW de 5 estaciones, 1/d² (caudal observado; 28/11) | −0,114 | 0,288 | 54 | 14 |
| Climatología de 5 donantes | −0,081 | −0,058 | 56 | 11 |
| Meta de la propuesta | 0,50 | — | — | — |

- **El LSTM sirve para cuencas sin estación.** El base supera a la climatología de donantes en 79 de 95 cuencas y al vecino en 83, con p ≈ 0 en las dos.
- **No llega a la meta de 0,50:** solo una de cada cuatro cuencas la alcanza.
- **Frente a LightGBM, empate** (51 de 95, p = 1,0).
- **IDW** (mejora B2, 28/11; `python src/baselines_espacial.py --solo-idw`): mejora al vecino, pero el LSTM base lo supera en 74 de 95 cuencas y el de ERA5-Land en 76 (p ≈ 0).
- **Costo de no tener estación** (62 cuencas, 2016–2022): de 0,425 a 0,248 en la variante base y de 0,480 a 0,242 con ERA5-Land. En las dos se pierde lo mismo en mediana por cuenca (−0,14).
- **ERA5-Land frente a base:** 0,300 frente a 0,276, con 54 de 95 cuencas mejorando, pero p = 0,08: es una tendencia. Con dos semillas daba 0,248, arrastrado por la semilla 43 (0,207). El ensamble es necesario.
- **Por región (base / ERA5-Land):** Orinoco 0,64 / 0,72; Caribe 0,26 / 0,29; Amazonía 0,23 / 0,29; Pacífico 0,15 / 0,16; Magdalena-Cauca −0,03 / −0,51 (solo 6 cuencas). En el Pacífico, LightGBM es mejor (0,36).
- **Errores:**
  - el modelo suaviza las crecientes: FHV −28 % y α = 0,67;
  - el volumen sale bien: β = 0,97;
  - las 22 cuencas con NSE < 0 están en Córdoba (cuenca del Sinú, 6), en el suroccidente (Nariño, Cauca y Putumayo, 7) y en la Sierra Nevada (4). Son más pequeñas que el resto (870 km² de área mediana, frente a 1420).

**Análisis de errores** (28/11; `src/errores_espacial.py`; notebook, sección 10.4):
- **Crecientes** (mejora B7), en 78 cuencas con al menos 10 años completos; umbrales de una Gumbel ajustada a los máximos anuales; un evento se detecta si el modelo tiene uno a ±2 días:
  - con el umbral propio de la simulación (lo único calculable sin estación), el LSTM con ERA5-Land detecta el 15 % de los eventos de 2 años (F1 = 0,16) y el 10 % de los de 5 años;
  - con el umbral observado, solo el 4 %;
  - el vecino más cercano detecta el 12 % con el umbral observado, pero con una precisión del 9 %;
  - con ±7 días, el F1 del LSTM sube a 0,26; con ±15 días, a 0,34.
  - **Conclusión: el modelo no sirve para alertas de creciente en cuencas sin estación.**
- **Residuos** (mejora B8): la correlación mediana de los errores diarios entre cuencas del mismo sistema de ríos a menos de 50 km es de 0,38; entre sistemas distintos a la misma distancia, 0,16; después de 100 km, casi 0. LightGBM muestra lo mismo. Los errores son regionales: las 95 cuencas no son pruebas independientes y los p de Wilcoxon son optimistas.

**Decisiones.**
- **El modelo final sigue siendo el LSTM con ERA5-Land.** Se eligió en la etapa 1 (sección 22) y la etapa 2 no lo contradice.
- El promedio con LightGBM no se elige: su ventaja (59 de 95 cuencas, p = 0,003) se observó mirando las cuencas retenidas, y la mediana casi no cambia.

**En curso (28/11):**
- el Transformer de NeuralHydrology con las entradas de ERA5-Land, semilla 42, 5 particiones (mejora B3 del plan);
- el despliegue en el visor Caudal-COL de Sebastián (plan, fase C).

---

## Registro de corridas

| Carpeta en `runs/` | Qué es | Épocas (mejor) | NSE prueba |
|---|---|---|---|
| `camels_col_humo_1011_165655` | Prueba de humo, lote 64 | 2 | — |
| `camels_col_humo_b256_1111_135228` | Prueba de humo, lote 256 | 2 | — |
| `camels_col_temporal_h256_eaa981_1111_171248` | Flujo temporal, lote 256 | 14 (3) | 0,394 |
| `camels_col_temporal_h256_b64_ffa963_1111_222122` | Flujo temporal, lote 64 | 11 (3) | 0,394 |
| `camels_col_temporal_h256_1981_2015_129122_1211_224752` | Flujo temporal, entrenado con 1981–2015 | 3 fijas | 0,398 |
| `camels_col_lightgbm_temporal_3e7a50` | LightGBM, flujo temporal | 783 árboles | 0,390 |
| `camels_col_temporal_relleno_1711_045443` | Flujo temporal con huecos rellenados | 12 (7) | 0,380 |
| `confirmacion_base_s{42,43,44}_*` | Confirmación, configuración actual (C1), 3 semillas | 3 fijas | — (validación 0,452) |
| `confirmacion_era5l_s{42,43,44}_*` | Confirmación, con ERA5-Land (C2), 3 semillas | 3 fijas | 0,432 en ensamble (validación 0,491) |
| `prueba_final_1911` | LightGBM con ERA5-Land, 3 semillas, y promedio final con C2 (sección 22) | 800 árboles fijos | 0,417 / **0,437** |
| (sin carpeta: script aparte) | Piloto de ERA5 con LightGBM, 30 cuencas, 2001–2010 | 500 árboles por bloque | 0,490 / 0,493 / 0,378 |
| `espacial_p{1..5}_s{42,43,44}_*` | Etapa 2, variante base, 15 corridas (sección 24) | 3 fijas | 0,276 en ensamble (95 cuencas no vistas, 1981–2022) |
| `espacial_era5l_p{1..5}_s{42,43,44}_*` | Etapa 2 con ERA5-Land, 15 corridas (sección 24) | 3 fijas | 0,300 en ensamble |
| `espacial_transformer_p{1..5}_s42_*` | Etapa 2, Transformer con las entradas de ERA5-Land (sección 24) | 3 fijas | en curso |
| `runs/flujo_espacial/baselines.parquet` | Referencias de la etapa 2: donantes, vecino y LightGBM (sección 24) | — | −0,081 / −0,426 / 0,257 |

---

## Registro de variables por iteración

Tabla viva: qué variables entraron en cada iteración, de dónde salen y cómo se calculan, para comparar iteraciones y escoger al final las variables más útiles. **La tabla completa, con una fila por variable (las 48 estáticas por separado), está en `docs/registro-variables.csv`.** Al agregar una iteración, se agrega una columna en los dos lugares.

**Desde el 18/11 el CSV tiene dos columnas más con el efecto medido en validación 2011–2015.** Las dos son la diferencia mediana por cuenca del NSE.
- `efecto_lgbm_nse_validacion`: quitar o agregar la variable o su grupo y reentrenar LightGBM (sección 19).
- `permutacion_lstm_nse_validacion`: revolverla en el LSTM ya entrenado (sección 20).

**Iteraciones**

| Código | Fecha | Modelo | Qué cambió | Evaluación | NSE mediano |
|---|---|---|---|---|---|
| H | 10–11/11 | LSTM | Prueba de humo | 20 cuencas, 2 épocas | — |
| T1 | 11/11 | LSTM | Configuración base, lote 256 | 183 cuencas, 2016–2022 | 0,394 |
| T2 | 12/11 | LSTM | Lote 64 | igual | 0,394 |
| T3 | 12/11 | LSTM | Entrenado con 1981–2015, 3 épocas fijas | igual | 0,398 |
| T4 | 16/11 | LightGBM | 35 variables derivadas de las forzantes, más los atributos | igual | 0,390 |
| T5 | 16/11 | Promedio de T1 y T4 | — | igual | **0,410** |
| T6 | 17/11 | LSTM | Base rellenada: lluvia de ClimateSERV y huecos largos con climatología | igual | 0,380 |
| T7 | 17/11 | LightGBM | Base rellenada | igual | 0,392 |
| T8 | 17/11 | Promedio de T6 y T7 | — | igual | 0,404 |
| P1a, P1b, P1c | 17/11 | LightGBM | Piloto de ERA5: nuestra lluvia, nuestra + ERA5, solo ERA5 (sección 14) | 30 cuencas, 2001–2010, validación cruzada | 0,490 / 0,493 / 0,378 |
| C1 | 18–19/11 | LSTM | Configuración de T1, con 3 semillas y 3 épocas fijas (sección 21) | 195 cuencas, validación 2011–2015, época 3 | 0,452 (ensamble 0,468) |
| C2 | 18–19/11 | LSTM | C1 más las 15 variables de ERA5-Land, sin `eroded_misce_perc` (sección 21) | igual | **0,491 (ensamble 0,505)** |
| T9 | 19/11 | LSTM | C2 en ensamble de 3 semillas, en la prueba (sección 22) | 183 cuencas, 2016–2022 | 0,432 |
| T10 | 19/11 | LightGBM | Base + ERA5-Land − `eroded_misce_perc` (116 variables), 800 árboles, 3 semillas | igual | 0,417 |
| T11 | 19/11 | Promedio de T9 y T10 | **Modelo final** | igual | **0,437** |

**Variables usadas en cada iteración** (✓ usada; R: reemplazada por la lluvia de ClimateSERV; c: huecos largos rellenados con climatología; E: ERA5 en lugar de `pr`)

| Variable | H | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | P1a | P1b | P1c | C1 | C2 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pr`: lluvia de CAMELS-COL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | R | R | ✓ | ✓ | E | ✓ | ✓ | ✓ | ✓ | ✓ |
| Lluvia CHIRPS de ClimateSERV | — | — | — | — | — | — | ✓ | ✓ | ✓ | — | — | — | — | — | — | — | — |
| Lluvia ERA5 (Open-Meteo) | — | — | — | — | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — | — |
| 15 variables de ERA5-Land (sección 17) | — | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | ✓ | d | ✓ |
| `poten_evapo` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | c | c | c | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `t_min`, `t_max` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | c | c | c | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 35 variables derivadas de las forzantes | — | — | — | — | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ |
| 48 atributos estáticos | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 47 | 47 | 47 | 47 |
| `streamflow` (solo objetivo) | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. | obj. |

"47": sin `eroded_misce_perc`. "d" en T10: las 15 variables de ERA5-Land entran como 34 columnas derivadas (sección 19): la del día, medias móviles, índices antecedentes, agua total del suelo y déficit de presión de vapor.

**Fuente, cálculo e importancia.** La importancia es el % de la ganancia total de LightGBM (T4) que aporta cada variable, sumando sus derivadas.

**Cómo se calcula `importancia_T4_pct`:**
1. Cada uno de los 783 árboles de T4 (hasta 63 hojas cada uno) divide los datos con preguntas del tipo "¿índice de lluvia antecedente de 20 días > 4 mm?".
2. Cada división reduce el error cuadrático ponderado del entrenamiento (con los pesos del NSE*). Esa reducción es la **ganancia** de la división.
3. Se suma la ganancia de todas las divisiones que usan cada columna, en todos los árboles: `feature_importance("gain")` de LightGBM.
4. Se expresa como % del total.
5. Se suman las columnas derivadas de una misma variable; por ejemplo, las 10 medias móviles de la lluvia quedan en `pr_media_2..365`.

Se calcula con los datos de entrenamiento, así que mide cuánto se apoya el modelo en cada variable, no cuánto mejora la predicción en prueba.

| Variable | Fuente | Cómo se calcula | Importancia en T4 |
|---|---|---|---|
| `pr` (mm/día) | CHIRPS v2.0 (UCSB): satélite infrarrojo más estaciones, celdas de 0,05° | CAMELS-COL: promedio simple de las celdas dentro de la cuenca, día a día. Equipo: interpolación lineal de huecos de hasta 7 días | **61,6 %** (43,8 % solo el índice de lluvia antecedente de 20 días) |
| `poten_evapo` (mm/día) | CAMELS-COL, a partir de las temperaturas de MSWX. Corregida en el pre-filtrado (media de 8,9 a 3,3 mm/día) | Hargreaves: 0,0023 × Ra × (T media + 17,8) × √(T máx − T mín), con Ra según latitud y día del año (FAO-56). **Comprobado el 17/11: coincide con nuestro propio cálculo con r = 1,000** | 2,4 % (más 2,5 % del balance lluvia − evapotranspiración) |
| `t_min`, `t_max` (°C) | MSWX (GloH2O; Beck et al., 2022): ERA5 corregido de sesgo y llevado a 0,1° | CAMELS-COL: promedio simple de las celdas dentro de la cuenca. Equipo: interpolación de huecos de hasta 7 días | 0,9 % |
| Lluvia CHIRPS de ClimateSERV | CHIRPS v2.0 por la API de ClimateSERV (NASA SERVIR) | Promedio que calcula el servidor dentro del polígono, con los polígonos corridos por el error de proyección (sección 9). **Descartada el 17/11** | — |
| Lluvia ERA5 | ERA5 (ECMWF), celdas de 0,25°, por la API de Open-Meteo | Promedio ponderado por la fracción de cada celda dentro de la cuenca, día local (UTC−5). Solo para el piloto | — |
| 15 variables de ERA5-Land (`era5l_*`) | ERA5-Land (ECMWF, Copernicus CDS), celdas de 0,1°: modelo de suelo de ECMWF forzado con la meteorología de ERA5 | Promedio ponderado por el área de cada celda dentro de la cuenca, día local (sección 17). Lluvia, radiación solar y térmica, evaporación real, escorrentía del modelo (total, superficial y subsuperficial; no es caudal observado), humedad del suelo en 4 capas, rocío, presión de vapor, viento y presión | — (no estaban en T4; en el LSTM suben el NSE de validación +0,024 por cuenca, sección 21) |
| Relleno climatológico | Equipo | Huecos de más de 7 días de evapotranspiración y temperaturas: promedio de ese mismo día del año en 1981–2010, suavizado con 31 días | — |
| Variables derivadas | Equipo (sección 8.3) | Lluvia de 1 a 3 días antes; medias de lluvia de 2 a 365 días; índices de lluvia antecedente (vida media de 5 a 180 días); medias de evapotranspiración, balance y temperaturas; día del año (seno y coseno, 1,1 %) | incluida arriba |
| 10 fisiográficos | CAMELS-COL: DEM SRTM de 30 m (IGAC) | Área (en log10, calculado por el equipo), forma, elevaciones, longitud y pendiente del cauce, número de curva, tiempo de concentración | 5,6 % |
| 7 climáticos | Equipo, con `pr` y `poten_evapo` de 1981–2010 | Medias, aridez, frecuencia y duración de días muy lluviosos y secos | 9,4 % |
| 5 de cobertura | CAMELS-COL: MapBiomas Colombia 2022 (Landsat, 30 m) | % del área de la cuenca en cada cobertura | 3,0 % |
| 6 de geología | CAMELS-COL: Mapa Geológico de Colombia 2023 (SGC) | % del área en cada tipo de roca | 3,1 % |
| 13 de suelos | CAMELS-COL: mapas de suelos del IGAC (1:100 000) | % del área en cada orden de suelo y en zonas urbanas, misceláneas y cuerpos de agua | 8,9 % (6,3 % solo `eroded_misce_perc`) |
| 7 de capacidad de uso | CAMELS-COL: capacidad de uso de las tierras del IGAC (1:100 000) | % del área en cada clase agrológica (2 a 8) | 2,0 % |
| `streamflow` (objetivo) | IDEAM, plataforma DHIME | m³/s convertidos a mm/día (Q × 86,4 / área), con el control de calidad del pre-filtrado. Nunca es entrada ni se rellena | — |

**Lectura (17/11):**
- **La lluvia acumulada lo es casi todo** en LightGBM: 62 % de la ganancia, y la mayor parte viene de un solo resumen, la lluvia de las últimas semanas. Los 48 atributos suman 32 %; la evapotranspiración, las temperaturas y el día del año, 6 %.
- **`poten_evapo` no es información independiente:** se calcula con `t_min`, `t_max` y la fecha. Lo nuevo que aporta frente a las temperaturas es solo la radiación teórica según el día del año.
- **`eroded_misce_perc` (6,3 %) es sospechoso:** solo 39 de las 248 cuencas tienen suelo erosionado, y solo 16 más del 1 %. El modelo puede estar usándolo para reconocer cuencas concretas.
  - Otros atributos igual de escasos casi no se usan. Por ejemplo, `unconso_depo_perc` tiene valor en 44 cuencas y pesa 0,02 %.
- **Limitación:** la ganancia mide cuánto usa el modelo una variable, no si es indispensable. Las variables parecidas se reparten la importancia. Para escoger variables hace falta quitarlas y reentrenar (pruebas de ablación con LightGBM, unos 5 minutos cada una).

**Selección provisional, antes de la ablación (17/11).** Mejor modelo hasta ahora: T5, el promedio del LSTM y LightGBM, con NSE 0,410.
- **Se quedan:**
  - `pr` (CHIRPS) y, en LightGBM, sus acumulados e índices antecedentes;
  - `t_min` y `t_max`;
  - los 7 atributos climáticos, área, pendiente, tiempo de concentración y cobertura.
- **Por probar, se quedan mientras tanto:**
  - `poten_evapo`, que es redundante con las temperaturas;
  - `eroded_misce_perc`;
  - los grupos de suelos, geología y capacidad de uso, con unos 25 atributos que pesan menos de 0,3 % cada uno.
- **Fuera:** la lluvia de ClimateSERV y el relleno climatológico.
- **En espera:** la lluvia de ERA5, que suma +0,01 en el piloto; solo entraría si se descarga ERA5-Land por la humedad del suelo.
