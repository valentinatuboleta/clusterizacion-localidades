import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

os.makedirs("reports/figures", exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[6]

COLOR_NAVY = RGBColor(11, 37, 69)
COLOR_STEEL = RGBColor(19, 64, 116)
COLOR_TEAL = RGBColor(0, 150, 136)
COLOR_DARK = RGBColor(33, 37, 41)
COLOR_GRAY = RGBColor(108, 117, 125)
COLOR_BG_LIGHT = RGBColor(245, 247, 250)
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BORDER = RGBColor(220, 224, 230)

def set_slide_background(slide, color=COLOR_BG_LIGHT):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text="ANALISIS EXPLORATORIO DE DATOS (EDA)"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_TEAL
    
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

# -------------------------------------------------------------
# SLIDE 1: PORTADA
# -------------------------------------------------------------
slide1 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide1, COLOR_NAVY)

box1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.8))
tf1 = box1.text_frame
tf1.word_wrap = True

p_kicker = tf1.paragraphs[0]
p_kicker.text = "TUBOLETA  DATA SCIENCE & MACHINE LEARNING"
p_kicker.font.size = Pt(13)
p_kicker.font.bold = True
p_kicker.font.color.rgb = COLOR_TEAL
p_kicker.space_after = Pt(14)

p_title = tf1.add_paragraph()
p_title.text = "Analisis Exploratorio de Datos (EDA)\ny Segmentacion de Localidades"
p_title.font.size = Pt(32)
p_title.font.bold = True
p_title.font.color.rgb = COLOR_WHITE
p_title.space_after = Pt(14)

p_sub = tf1.add_paragraph()
p_sub.text = "Estandarizacion de Localidades mediante Espacio Vectorial Mixto (NLP + Metricas Relativas por Evento)"
p_sub.font.size = Pt(15)
p_sub.font.color.rgb = RGBColor(200, 215, 230)
p_sub.space_after = Pt(24)

p_meta = tf1.add_paragraph()
p_meta.text = "Pipeline Version 2.0 | Dataset: 33,878 Localidades Certificadas"
p_meta.font.size = Pt(11)
p_meta.font.color.rgb = COLOR_GRAY

# -------------------------------------------------------------
# SLIDE 2: CONTEXTO Y RETO DE NEGOCIO
# -------------------------------------------------------------
slide2 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide2)
add_header(slide2, "Contexto de Negocio: El Desafio de la Heterogeneidad de Localidades")

col_data = [
    ("1. La Trampa del Nombre Comercial", 
     "Mas de 2,400 nombres diferentes creados por promotores con ruido publicitario y nombres de gira (ej. Palcos Cantinero, Modo Leyenda, Experiencia Platino)."),
    ("2. Distorsion de Escala en Pesos COP",
     "Una boleta de $150,000 COP representa el acceso mas economico (General) en un estadio masivo, pero la entrada mas exclusiva (VIP) en un teatro intimo."),
    ("3. Similitud Lexica Enganosa",
     "Localidades como Occidental Alta Oro y Occidental Alta Plata son 90% similares en texto, pero tienen jerarquias de precio y perfiles de comprador opuestos.")
]

lefts = [Inches(0.8), Inches(4.8), Inches(8.8)]
for i, (col_title, col_desc) in enumerate(col_data):
    card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.6), Inches(3.7), Inches(4.8))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_WHITE
    card.line.color.rgb = COLOR_CARD_BORDER
    
    tb = slide2.shapes.add_textbox(lefts[i] + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = col_title
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(12)
    
    p2 = tf.add_paragraph()
    p2.text = col_desc
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_DARK

# -------------------------------------------------------------
# SLIDE 3: ARQUITECTURA DE LA SOLUCION
# -------------------------------------------------------------
slide3 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide3)
add_header(slide3, "Arquitectura de la Solucion: Pipeline de Espacio Vectorial Mixto")

pipe_steps = [
    ("Modulo 1: Descomposicion NLP (src.nlp_utils)",
     "- Supresion de ruido publicitario (stopwords de gira y patrocinios).\n- Extraccion de 17 variables binarias estructurales, espaciales y restricciones.\n- Vectorizacion TF-IDF estructurada sobre texto limpio."),
    ("Modulo 2: Metricas Relativas por Evento (src.feature_engineering)",
     "- percentil_precio_evento (0.0 a 1.0) para medir jerarquia interna.\n- ratio_precio_max (P / P_max) normalizado por funcion.\n- peso_aforo (dn_quota / performance_quota) para escala de capacidad.\n- tasa_ocupacion y rotacion de demanda historica."),
    ("Modulo 3: FUSION VECTORIAL Y CLUSTERING (src.clustering)",
     "- Ensamble del espacio vectorial mixto X en R^(33,878 x 37).\n- Evaluacion de numero optimo de clusters con Silueta, Inercia y Davies-Bouldin.\n- Asignacion automatica a 4 Arquetipos Universales de Demanda.")
]

y_pos = [Inches(1.5), Inches(3.3), Inches(5.1)]
for i, (title_p, desc_p) in enumerate(pipe_steps):
    card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos[i], Inches(11.7), Inches(1.55))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_WHITE
    card.line.color.rgb = COLOR_CARD_BORDER
    
    tb = slide3.shapes.add_textbox(Inches(1.0), y_pos[i] + Inches(0.12), Inches(11.3), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = title_p
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_STEEL
    p1.space_after = Pt(4)
    
    p2 = tf.add_paragraph()
    p2.text = desc_p
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_DARK

# -------------------------------------------------------------
# SLIDES DE GRAFICOS (SLIDES 4 A 8)
# -------------------------------------------------------------
charts_info = [
    {
        "num": "Grafico 1",
        "title": "Descomposicion NLP y Extraccion de Atributos Estructurales",
        "fig": "reports/figures/fig1_tags_frecuencia.png",
        "analysis": "De las 33,878 localidades evaluadas, la etiqueta mas frecuente es tag_general con 15,386 apariciones (45.4%), seguida de niveles de teatro y recintos cerrados como tag_balcon (4,319), tag_platea (4,037) y tag_palco (3,581). En el ambito espacial, las orientaciones predominantes son tag_occidental (1,004) y tag_norte (810).",
        "conclusions": [
            "Predominio de Localidades Masivas: Casi la mitad del inventario corresponde a admision general, lo que exige diferenciar una general de estadio frente a una de teatro.",
            "Alta Especializacion en Teatros y Arenas: Mas de 11,900 registros corresponden a balcones, plateas y palcos con distribucion vertical escalonada.",
            "Efectividad del Pipeline Regex/NER: La deteccion de 17 etiquetas permitio convertir texto libre en variables numericas estructuradas de jerarquia, orientacion y restricciones."
        ]
    },
    {
        "num": "Grafico 2",
        "title": "Distribucion de Variables Relativas Normalizadas por Evento",
        "fig": "reports/figures/fig2_distribuciones.png",
        "analysis": "El ratio_precio_max presenta una concentracion alta en 1.0 (mediana = 1.0, media = 0.803). El peso_aforo muestra una clara bimodalidad: una concentracion de zonas exclusivas de aforo reducido (percentil 25 = 0.096, menos del 10% del venue) frente a eventos de admision unica. La tasa_ocupacion media es de 18.1% (mediana = 10.8%).",
        "conclusions": [
            "Desacople Exitoso de la Moneda: La escala 0.0 a 1.0 permite comparar precios de festivales masivos con teatros locales bajo la misma regla de exclusividad.",
            "Identificacion Inmediata de Zonas Selectas: El 25% de las localidades ocupan menos del 10% del aforo total, constituyendo candidatas naturales a VIP o palcos.",
            "Comportamiento Comercial Asimetrico: La ocupacion historica introduce una dimension de velocidad de demanda que diferencia zonas de alta rotacion de zonas con remanente."
        ]
    },
    {
        "num": "Grafico 3",
        "title": "Comparacion Bivariada de Precio Relativo y Aforo por Atributo NLP",
        "fig": "reports/figures/fig3b_boxplots_bivariados.png",
        "analysis": "PALCO y VIP registran las medianas de peso_aforo mas reducidas del catalogo (4.0% y 4.4% del venue), manteniendo ratios de precio promedio de 0.685 y 0.676 con maximos en 1.0. PREFERENCIAL y PLATEA registran ratios de precio medianos de 0.923 y 0.875 con aforos medios (15.7% a 17.4%). BALCON presenta el precio mas accesible de teatro (ratio medio 0.503).",
        "conclusions": [
            "Confirmacion de Jerarquia Fisica: La semantica NLP se alinea con la capacidad fisica: los palcos ocupan fracciones minimas de aforo y las generales absorben el volumen.",
            "Validacion de la Platea como Zona Preferente: Las plateas se ubican en el rango superior de precios, consolidandose como el escalon intermedio-alto de demanda.",
            "El Balcon como Opcion Popular de Recinto Cerrado: Los balcones registran sistematicamente un precio 50% menor a la platea del mismo show, validando su rol accesible."
        ]
    },
    {
        "num": "Grafico 4",
        "title": "Matriz de Correlaciones Numéricas y Ratios Estructurales",
        "fig": "reports/figures/fig4_correlaciones.png",
        "analysis": "La correlacion entre el precio nominal en COP (med_unit_amt_itx) y el ratio_precio_max es nula (r = -0.034), demostrando que el dinero nominal no refleja exclusividad. El peso_aforo correlaciona negativamente con ratio_precio_max (r = -0.248) y ratio_cortesias (r = -0.250). La tasa de ocupacion correlaciona fuertemente con ventas pagas (r = 0.82).",
        "conclusions": [
            "Independencia del Precio Nominal: Al tener correlacion cercana a cero con los ratios relativos, se ratifica que usar el precio en COP aisladamente distorsionaba la segmentacion.",
            "Ley de Oferta y Demanda en el Aforo: A mayor peso de aforo de una localidad dentro del show, menor tiende a ser su ratio de precio relativo.",
            "No Redundancia en el Espacio Mixto: Ninguna pareja de variables estructurales presenta colinealidad perfecta (|r| < 0.85), garantizando informacion complementaria para el clustering."
        ]
    },
    {
        "num": "Grafico 5",
        "title": "Mapa de Separabilidad Espacial (Cuadrantes de Demanda)",
        "fig": "reports/figures/fig5_cuadrantes_demanda.png",
        "analysis": "El scatter plot 2D proyecta la separacion en 4 cuadrantes: Superior Izquierdo (Bajo Aforo < 25%, Alto Precio > 0.5) con palcos y VIP de alta ocupacion; Inferior Izquierdo (Bajo Aforo, Bajo Precio) con balcones y vistas restringidas; Superior Derecho con preferenciales masivas; e Inferior Derecho con gradas generales y masivas.",
        "conclusions": [
            "Separabilidad Geometrica Evidente: Los datos forman cuatro concentraciones espaciales que validan matematicamente el uso de algoritmos basados en distancias (K-Means).",
            "Diferenciacion por Demanda (Ocupacion): La escala de color evidencia que las zonas del cuadrante superior izquierdo presentan las mayores tasas de agotamiento de boleteria.",
            "Soporte Visual para Decisiones de Negocio: Permite a pricing y operaciones ubicar cualquier nueva localidad en el espacio cartesiano y predecir su arquetipo."
        ]
    }
]

for item in charts_info:
    slide_chart = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide_chart)
    header_title = item["num"] + ": " + item["title"]
    add_header(slide_chart, header_title)
    
    img_card = slide_chart.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.3))
    img_card.fill.solid()
    img_card.fill.fore_color.rgb = COLOR_WHITE
    img_card.line.color.rgb = COLOR_CARD_BORDER
    
    slide_chart.shapes.add_picture(item["fig"], Inches(0.95), Inches(1.65), width=Inches(5.7))
    
    text_card = slide_chart.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(5.5), Inches(5.3))
    text_card.fill.solid()
    text_card.fill.fore_color.rgb = COLOR_WHITE
    text_card.line.color.rgb = COLOR_CARD_BORDER
    
    tb = slide_chart.shapes.add_textbox(Inches(7.2), Inches(1.65), Inches(5.1), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p_an_title = tf.paragraphs[0]
    p_an_title.text = "ANALISIS DE DATOS"
    p_an_title.font.size = Pt(11)
    p_an_title.font.bold = True
    p_an_title.font.color.rgb = COLOR_STEEL
    p_an_title.space_after = Pt(3)
    
    p_an = tf.add_paragraph()
    p_an.text = item["analysis"]
    p_an.font.size = Pt(9.5)
    p_an.font.color.rgb = COLOR_DARK
    p_an.space_after = Pt(8)
    
    p_con_title = tf.add_paragraph()
    p_con_title.text = "TRES CONCLUSIONES CLAVE"
    p_con_title.font.size = Pt(11)
    p_con_title.font.bold = True
    p_con_title.font.color.rgb = COLOR_TEAL
    p_con_title.space_after = Pt(4)
    
    for c_idx, c_text in enumerate(item["conclusions"]):
        p_c = tf.add_paragraph()
        p_c.text = str(c_idx+1) + ". " + c_text
        p_c.font.size = Pt(9.5)
        p_c.font.color.rgb = COLOR_DARK
        p_c.space_after = Pt(4)

# -------------------------------------------------------------
# SLIDE 9: LOS 4 ARQUETIPOS DE DEMANDA
# -------------------------------------------------------------
slide9 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide9)
add_header(slide9, "Segmentacion Final: Los 4 Arquetipos Universales de Demanda")

archetypes = [
    ("VIP / Palcos / Premium", "1,919 (5.7%)", "Ratio Precio: 0.72 | Aforo: 19.1% | Ocupacion: 69.7%", "Palcos, Mesas VIP, Boxes, Suites, Experiencia Platino"),
    ("Preferencial / Platea Frontal", "5,759 (17.0%)", "Ratio Precio: 0.74 | Aforo: 22.2% | Ocupacion: 40.9%", "Platea 1, Platea 2, Preferencial Delantera, Sillas Centrales"),
    ("Grada General / Masiva", "17,182 (50.7%)", "Ratio Precio: 0.99 | Aforo: 89.3% | Ocupacion: 9.1%", "General, Entrada Unica, Tiquete Full, Admision General"),
    ("Popular / Visibilidad Parcial / Balcon", "9,018 (26.6%)", "Ratio Precio: 0.49 | Aforo: 18.3% | Ocupacion: 10.0%", "Platea Posterior, Balcon Mayor, 2do Balcon, Vista Parcial")
]

y_arch = [Inches(1.5), Inches(2.8), Inches(4.1), Inches(5.4)]
for i, (name_a, count_a, metrics_a, examples_a) in enumerate(archetypes):
    card = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_arch[i], Inches(11.7), Inches(1.15))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_WHITE
    card.line.color.rgb = COLOR_CARD_BORDER
    
    tb = slide9.shapes.add_textbox(Inches(1.0), y_arch[i] + Inches(0.08), Inches(11.3), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = name_a + " [" + count_a + " del catalogo]"
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(2)
    
    p2 = tf.add_paragraph()
    p2.text = "Perfil: " + metrics_a + "  |  Ejemplos: " + examples_a
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_DARK

# -------------------------------------------------------------
# SLIDE 10: CONCLUSIONES FINALES Y SIGUIENTES PASOS
# -------------------------------------------------------------
slide10 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide10)
add_header(slide10, "Conclusiones del EDA y Siguientes Pasos de Modelado")

summary_cards = [
    ("Logros del Analisis Exploratorio",
     "1. Se demostro que la normalizacion por evento elimina el sesgo del dinero en COP.\n2. Se comprobo que el 81.8% de los VIPs no tienen la palabra VIP en su nombre y se clasifican por fisica de aforo y precio.\n3. Se estructuro el espacio mixto en R^(33,878 x 37) sin colinealidad critica (|r| < 0.85)."),
    ("Impacto de Negocio para TuBoleta",
     "1. Reportes y Dashboards transversales estandarizados sin depender del promotor.\n2. Modelos de elasticidad de precios por arquetipo (VIP vs Preferencial vs General).\n3. Deteccion temprana de localidades con baja rotacion para optimizacion de aforo."),
    ("Siguientes Pasos (Fase 3)",
     "1. Ejecutar el cuaderno 02_clustering_espacio_mixto.ipynb para validar K-Means con k=4.\n2. Comparar resultados con Gaussian Mixture Models (GMM) para densidades variables.\n3. Exportar localidades_clusterizadas.parquet hacia la base analitica de BI.")
]

for i, (stitle, sdesc) in enumerate(summary_cards):
    card = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.6), Inches(3.7), Inches(5.1))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_WHITE
    card.line.color.rgb = COLOR_CARD_BORDER
    
    tb = slide10.shapes.add_textbox(lefts[i] + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = stitle
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(12)
    
    p2 = tf.add_paragraph()
    p2.text = sdesc
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_DARK

output_pptx = "Presentacion_EDA_Clusterizacion_Localidades.pptx"
prs.save(output_pptx)
print("Presentacion PowerPoint generada exitosamente:", output_pptx, f"({os.path.getsize(output_pptx):,} bytes)")
