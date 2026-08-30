import os
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation_from_template(
    template_path="template.pptx",
    output_path="Presentacion_EDA_Clusterizacion_Localidades.pptx"
):
    prs = Presentation(template_path)
    
    # slide 0: Portada
    # slide 1: Slide genérica base del template
    template_cover = prs.slides[0]
    template_generic = prs.slides[1]
    
    # Paleta de colores corporativos
    COLOR_NAVY = RGBColor(11, 37, 69)       # #0B2545
    COLOR_STEEL = RGBColor(19, 64, 116)     # #134074
    COLOR_TEAL = RGBColor(0, 150, 136)      # #009688
    COLOR_DARK = RGBColor(33, 37, 41)       # #212529
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_CARD_BORDER = RGBColor(215, 220, 228)
    COLOR_CARD_BG = RGBColor(255, 255, 255)
    
    FONT_NAME = "Montserrat"
    
    # -------------------------------------------------------------
    # 1. SLIDE 1: PORTADA
    # -------------------------------------------------------------
    for shape in template_cover.shapes:
        if shape.has_text_frame:
            if "Titulo" in shape.text:
                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = "Clusterizacion y Estandarizacion\nde Localidades"
                p.font.name = FONT_NAME
                p.font.size = Pt(28)
                p.font.bold = True
                p.font.color.rgb = COLOR_NAVY
                
                shape.left = Inches(1.2)
                shape.top = Inches(2.2)
                shape.width = Inches(8.5)
                shape.height = Inches(1.5)
            elif "Equipo BI" in shape.text:
                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = "Analisis Exploratorio de Datos (EDA) | Equipo BI & Data Science"
                p.font.name = FONT_NAME
                p.font.size = Pt(14)
                p.font.bold = False
                p.font.color.rgb = COLOR_STEEL
                
                shape.left = Inches(1.2)
                shape.top = Inches(3.8)
                shape.width = Inches(8.5)
                shape.height = Inches(0.6)

    # Helper para crear una nueva diapositiva clonando exactamente la plantilla de la Slide 2
    def create_slide_from_template_generic(tema_text, desc_text):
        # Crear nueva diapositiva con el layout de la diapositiva 2
        new_slide = prs.slides.add_slide(template_generic.slide_layout)
        
        # 1. Vincular la imagen corporativa de fondo en las relaciones (.rels) de la nueva slide
        img_part = template_generic.part.related_part("rId3")
        new_rId = new_slide.part.relate_to(
            img_part,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        
        # 2. Copiar el elemento <p:bg> exacto y actualizar el blip r:embed con new_rId
        bg_elem = copy.deepcopy(template_generic._element.xpath(".//p:bg")[0])
        blip = bg_elem.xpath(".//a:blip")[0]
        blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", new_rId)
        
        cSld = new_slide._element.xpath(".//p:cSld")[0]
        if len(cSld.xpath("./p:bg")) > 0:
            cSld.replace(cSld.xpath("./p:bg")[0], bg_elem)
        else:
            cSld.insert(0, bg_elem)
            
        # 3. Eliminar formas predeterminadas del layout para construir contenido limpio
        for sh in list(new_slide.shapes):
            sp = sh._element
            sp.getparent().remove(sp)
            
        # 4. Cuadro de Tema (Encabezado Superior sobre la barra) - Montserrat 18pt Negrita Blanco
        tb_tema = new_slide.shapes.add_textbox(Inches(0.92), Inches(0.19), Inches(8.16), Inches(0.40))
        tf_tema = tb_tema.text_frame
        tf_tema.word_wrap = True
        p_tema = tf_tema.paragraphs[0]
        p_tema.text = tema_text.upper()
        p_tema.font.name = FONT_NAME
        p_tema.font.size = Pt(18)
        p_tema.font.bold = True
        p_tema.font.color.rgb = COLOR_WHITE
        
        # 5. Cuadro de Descripción (Subtítulo de Sección) - Montserrat 14pt Negrita Steel Blue
        tb_desc = new_slide.shapes.add_textbox(Inches(0.45), Inches(1.30), Inches(11.62), Inches(0.40))
        tf_desc = tb_desc.text_frame
        tf_desc.word_wrap = True
        p_desc = tf_desc.paragraphs[0]
        p_desc.alignment = PP_ALIGN.CENTER
        p_desc.text = desc_text
        p_desc.font.name = FONT_NAME
        p_desc.font.size = Pt(14)
        p_desc.font.bold = True
        p_desc.font.color.rgb = COLOR_STEEL
        
        return new_slide

    # -------------------------------------------------------------
    # 2. SLIDE 2: CONTEXTO Y RETOS DE NEGOCIO (Usando la slide 2 existente)
    # -------------------------------------------------------------
    slide2 = template_generic
    # Limpiar formas default de slide 2
    for sh in list(slide2.shapes):
        sp = sh._element
        sp.getparent().remove(sp)
        
    # Encabezados de slide 2
    tb_tema2 = slide2.shapes.add_textbox(Inches(0.92), Inches(0.19), Inches(8.16), Inches(0.40))
    p_t2 = tb_tema2.text_frame.paragraphs[0]
    p_t2.text = "CONTEXTO DE NEGOCIO"
    p_t2.font.name = FONT_NAME
    p_t2.font.size = Pt(18)
    p_t2.font.bold = True
    p_t2.font.color.rgb = COLOR_WHITE
    
    tb_desc2 = slide2.shapes.add_textbox(Inches(0.45), Inches(1.30), Inches(11.62), Inches(0.40))
    p_d2 = tb_desc2.text_frame.paragraphs[0]
    p_d2.alignment = PP_ALIGN.CENTER
    p_d2.text = "Heterogeneidad de Localidades y Desafios en la Boleteria Tradicional"
    p_d2.font.name = FONT_NAME
    p_d2.font.size = Pt(14)
    p_d2.font.bold = True
    p_d2.font.color.rgb = COLOR_STEEL

    col_data = [
        ("1. La Trampa del Nombre Comercial", 
         "Mas de 2,400 nombres comerciales diferentes creados por promotores con ruido publicitario y nombres de gira (ej. Palcos Cantinero, Modo Leyenda, Experiencia Platino)."),
        ("2. Distorsion de Escala en Pesos COP",
         "Una boleta de $150,000 COP representa el acceso mas economico (General) en un estadio masivo, pero la entrada mas exclusiva (VIP) en un teatro intimo."),
        ("3. Similitud Lexica Enganosa",
         "Localidades como Occidental Alta Oro y Occidental Alta Plata son 90% similares en texto, pero tienen jerarquias de precio y perfiles de comprador opuestos.")
    ]
    lefts = [Inches(0.8), Inches(4.8), Inches(8.8)]
    for i, (col_title, col_desc) in enumerate(col_data):
        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.85), Inches(3.7), Inches(4.85))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide2.shapes.add_textbox(lefts[i] + Inches(0.2), Inches(2.05), Inches(3.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = col_title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_NAVY
        p1.space_after = Pt(12)
        
        p2 = tf.add_paragraph()
        p2.text = col_desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(13)
        p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # 3. SLIDE 3: ARQUITECTURA DE LA SOLUCION
    # -------------------------------------------------------------
    slide3 = create_slide_from_template_generic(
        "ARQUITECTURA DEL PIPELINE",
        "Enfoque Metodologico: Espacio Vectorial Mixto (NLP + Metricas Relativas)"
    )
    pipe_steps = [
        ("Modulo 1: Descomposicion NLP (src.nlp_utils)",
         "- Supresion de ruido publicitario (stopwords de gira y patrocinios).\n- Extraccion de 17 variables binarias estructurales, espaciales y restricciones.\n- Vectorizacion TF-IDF estructurada sobre texto limpio."),
        ("Modulo 2: Metricas Relativas por Evento (src.feature_engineering)",
         "- percentil_precio_evento (0.0 a 1.0) para medir jerarquia interna del evento.\n- ratio_precio_max (P / P_max) normalizado por funcion.\n- peso_aforo (dn_quota / performance_quota) para capturar escala de capacidad.\n- tasa_ocupacion y absorcion de demanda historica."),
        ("Modulo 3: FUSION VECTORIAL Y CLUSTERING (src.clustering)",
         "- Ensamble del espacio vectorial mixto X en R^(33,878 x 37).\n- Evaluacion de numero optimo de clusters con Silueta, Inercia y Davies-Bouldin.\n- Asignacion automatica a 4 Arquetipos Universales de Demanda.")
    ]
    y_pos = [Inches(1.85), Inches(3.55), Inches(5.25)]
    for i, (title_p, desc_p) in enumerate(pipe_steps):
        card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos[i], Inches(11.7), Inches(1.5))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide3.shapes.add_textbox(Inches(1.0), y_pos[i] + Inches(0.1), Inches(11.3), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = title_p
        p1.font.name = FONT_NAME
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_STEEL
        p1.space_after = Pt(3)
        
        p2 = tf.add_paragraph()
        p2.text = desc_p
        p2.font.name = FONT_NAME
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # SLIDES 4 A 8: GRAFICOS CON ANALISIS Y CONCLUSIONES
    # -------------------------------------------------------------
    charts_info = [
        {
            "tema": "MODULO 1: DESCOMPOSICION NLP",
            "desc": "Grafico 1: Frecuencia de Atributos Estructurales, Espaciales y Restricciones Extraidos",
            "fig": "reports/figures/fig1_tags_frecuencia.png",
            "analysis": "De las 33,878 localidades evaluadas, la etiqueta mas frecuente es tag_general con 15,386 apariciones (45.4%), seguida de niveles de teatro y recintos cerrados como tag_balcon (4,319), tag_platea (4,037) y tag_palco (3,581). En el ambito espacial, las orientaciones predominantes son tag_occidental (1,004) y tag_norte (810).",
            "conclusions": [
                "Predominio de Localidades Masivas: Casi la mitad del inventario corresponde a admision general, lo que exige diferenciar una general de estadio frente a una de teatro.",
                "Alta Especializacion en Teatros y Arenas: Mas de 11,900 registros corresponden a balcones, plateas y palcos con distribucion vertical escalonada.",
                "Efectividad del Pipeline Regex/NER: La deteccion de 17 etiquetas permitio convertir texto libre en variables numericas estructuradas de jerarquia, orientacion y restricciones."
            ]
        },
        {
            "tema": "MODULO 2: METRICAS RELATIVAS",
            "desc": "Grafico 2: Distribucion de Variables Relativas Normalizadas por Evento",
            "fig": "reports/figures/fig2_distribuciones.png",
            "analysis": "El ratio_precio_max presenta una concentracion alta en 1.0 (mediana = 1.0, media = 0.803). El peso_aforo muestra una clara bimodalidad: una concentracion de zonas exclusivas de aforo reducido (percentil 25 = 0.096, menos del 10% del venue) frente a eventos de admision unica. La tasa_ocupacion media es de 18.1% (mediana = 10.8%).",
            "conclusions": [
                "Desacople Exitoso de la Moneda: La escala 0.0 a 1.0 permite comparar precios de festivales masivos con teatros locales bajo la misma regla de exclusividad.",
                "Identificacion Inmediata de Zonas Selectas: El 25% de las localidades ocupan menos del 10% del aforo total, constituyendo candidatas naturales a VIP o palcos.",
                "Comportamiento Comercial Asimetrico: La ocupacion historica introduce una dimension de velocidad de demanda que diferencia zonas de alta rotacion de zonas con remanente."
            ]
        },
        {
            "tema": "VALIDACION SEMANTICA Y FISICA",
            "desc": "Grafico 3: Comparacion Bivariada de Precio Relativo y Aforo por Atributo NLP",
            "fig": "reports/figures/fig3_boxplots_bivariados.png",
            "analysis": "PALCO y VIP registran las medianas de peso_aforo mas reducidas del catalogo (4.0% y 4.4% del venue), manteniendo ratios de precio promedio de 0.685 y 0.676 con maximos en 1.0. PREFERENCIAL y PLATEA registran ratios de precio medianos de 0.923 y 0.875 con aforos medios (15.7% a 17.4%). BALCON presenta el precio mas accesible de teatro (ratio medio 0.503).",
            "conclusions": [
                "Confirmacion de Jerarquia Fisica: La semantica NLP se alinea con la capacidad fisica: los palcos ocupan fracciones minimas de aforo y las generales absorben el volumen.",
                "Validacion de la Platea como Zona Preferente: Las plateas se ubican en el rango superior de precios, consolidandose como el escalon intermedio-alto de demanda.",
                "El Balcon como Opcion Popular de Recinto Cerrado: Los balcones registran sistematicamente un precio 50% menor a la platea del mismo show, validando su rol accesible."
            ]
        },
        {
            "tema": "ANALISIS DE CORRELACIONES",
            "desc": "Grafico 4: Matriz de Correlaciones Numericas y Ratios Estructurales",
            "fig": "reports/figures/fig4_correlaciones.png",
            "analysis": "La correlacion entre el precio nominal en COP (med_unit_amt_itx) y el ratio_precio_max es nula (r = -0.034), demostrando que el dinero nominal no refleja exclusividad. El peso_aforo correlaciona negativamente con ratio_precio_max (r = -0.248) y ratio_cortesias (r = -0.250). La tasa de ocupacion correlaciona fuertemente con ventas pagas (r = 0.82).",
            "conclusions": [
                "Independencia del Precio Nominal: Al tener correlacion cercana a cero con los ratios relativos, se ratifica que usar el precio en COP aisladamente distorsionaba la segmentacion.",
                "Ley de Oferta y Demanda en el Aforo: A mayor peso de aforo de una localidad dentro del show, menor tiende a ser su ratio de precio relativo.",
                "No Redundancia en el Espacio Mixto: Ninguna pareja de variables estructurales presenta colinealidad perfecta (|r| < 0.85), garantizando informacion complementaria para el clustering."
            ]
        },
        {
            "tema": "ESPACIO DE CLUSTERING",
            "desc": "Grafico 5: Mapa de Separabilidad Espacial (Cuadrantes de Demanda)",
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
        slide_chart = create_slide_from_template_generic(item["tema"], item["desc"])
        
        # Tarjeta contenedor de imagen a la izquierda
        card_img = slide_chart.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(5.9), Inches(5.0))
        card_img.fill.solid()
        card_img.fill.fore_color.rgb = COLOR_CARD_BG
        card_img.line.color.rgb = COLOR_CARD_BORDER
        
        # Imagen del gráfico
        slide_chart.shapes.add_picture(item["fig"], Inches(0.95), Inches(1.95), width=Inches(5.6))
        
        # Tarjeta contenedor de texto a la derecha
        card_txt = slide_chart.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.85), Inches(5.6), Inches(5.0))
        card_txt.fill.solid()
        card_txt.fill.fore_color.rgb = COLOR_CARD_BG
        card_txt.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide_chart.shapes.add_textbox(Inches(7.1), Inches(1.95), Inches(5.2), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_an_title = tf.paragraphs[0]
        p_an_title.text = "ANALISIS DE DATOS"
        p_an_title.font.name = FONT_NAME
        p_an_title.font.size = Pt(14)
        p_an_title.font.bold = True
        p_an_title.font.color.rgb = COLOR_STEEL
        p_an_title.space_after = Pt(2)
        
        p_an = tf.add_paragraph()
        p_an.text = item["analysis"]
        p_an.font.name = FONT_NAME
        p_an.font.size = Pt(11)
        p_an.font.color.rgb = COLOR_DARK
        p_an.space_after = Pt(6)
        
        p_con_title = tf.add_paragraph()
        p_con_title.text = "TRES CONCLUSIONES CLAVE"
        p_con_title.font.name = FONT_NAME
        p_con_title.font.size = Pt(14)
        p_con_title.font.bold = True
        p_con_title.font.color.rgb = COLOR_TEAL
        p_con_title.space_after = Pt(3)
        
        for c_idx, c_text in enumerate(item["conclusions"]):
            p_c = tf.add_paragraph()
            p_c.text = str(c_idx+1) + ". " + c_text
            p_c.font.name = FONT_NAME
            p_c.font.size = Pt(11)
            p_c.font.color.rgb = COLOR_DARK
            p_c.space_after = Pt(3)

    # -------------------------------------------------------------
    # 9. SLIDE 9: LOS 4 ARQUETIPOS DE DEMANDA
    # -------------------------------------------------------------
    slide9 = create_slide_from_template_generic(
        "ARQUETIPOS DE DEMANDA",
        "Estandarizacion de 33,878 Localidades en 4 Segmentos Universales"
    )
    archetypes = [
        ("VIP / Palcos / Premium", "1,919 (5.7%)", "Ratio Precio: 0.72 | Aforo: 19.1% | Ocupacion: 69.7%", "Palcos, Mesas VIP, Boxes, Suites, Experiencia Platino"),
        ("Preferencial / Platea Frontal", "5,759 (17.0%)", "Ratio Precio: 0.74 | Aforo: 22.2% | Ocupacion: 40.9%", "Platea 1, Platea 2, Preferencial Delantera, Sillas Centrales"),
        ("Grada General / Masiva", "17,182 (50.7%)", "Ratio Precio: 0.99 | Aforo: 89.3% | Ocupacion: 9.1%", "General, Entrada Unica, Tiquete Full, Admision General"),
        ("Popular / Visibilidad Parcial / Balcon", "9,018 (26.6%)", "Ratio Precio: 0.49 | Aforo: 18.3% | Ocupacion: 10.0%", "Platea Posterior, Balcon Mayor, 2do Balcon, Vista Parcial")
    ]
    y_arch = [Inches(1.85), Inches(3.1), Inches(4.35), Inches(5.6)]
    for i, (name_a, count_a, metrics_a, examples_a) in enumerate(archetypes):
        card = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_arch[i], Inches(11.7), Inches(1.1))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide9.shapes.add_textbox(Inches(1.0), y_arch[i] + Inches(0.08), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = name_a + " [" + count_a + " del catalogo]"
        p1.font.name = FONT_NAME
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_NAVY
        p1.space_after = Pt(2)
        
        p2 = tf.add_paragraph()
        p2.text = "Perfil: " + metrics_a + "  |  Ejemplos: " + examples_a
        p2.font.name = FONT_NAME
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # 10. SLIDE 10: CONCLUSIONES Y SIGUIENTES PASOS
    # -------------------------------------------------------------
    slide10 = create_slide_from_template_generic(
        "SINTESIS Y PROXIMOS PASOS",
        "Impacto Estrategico para TuBoleta y Pase a Produccion"
    )
    summary_cards = [
        ("Logros del Analisis Exploratorio",
         "1. Se demostro que la normalizacion por evento elimina el sesgo del dinero en COP.\n2. Se comprobo que el 81.8% de los VIPs no tienen la palabra VIP en su nombre y se clasifican por fisica de aforo y precio.\n3. Se estructuro el espacio mixto en R^(33,878 x 37) sin colinealidad critica (|r| < 0.85)."),
        ("Impacto de Negocio para TuBoleta",
         "1. Reportes y Dashboards transversales estandarizados sin depender del promotor.\n2. Modelos de elasticidad de precios por arquetipo (VIP vs Preferencial vs General).\n3. Deteccion temprana de localidades con baja rotacion para optimizacion de aforo."),
        ("Siguientes Pasos en Produccion",
         "1. Validar K-Means con k=4 en el pipeline de ejecucion.\n2. Comparar resultados con Gaussian Mixture Models (GMM) para densidades variables.\n3. Exportar localidades_clusterizadas.parquet hacia la base analitica de BI.")
    ]
    for i, (stitle, sdesc) in enumerate(summary_cards):
        card = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.85), Inches(3.7), Inches(4.85))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide10.shapes.add_textbox(lefts[i] + Inches(0.2), Inches(2.05), Inches(3.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = stitle
        p1.font.name = FONT_NAME
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_NAVY
        p1.space_after = Pt(10)
        
        p2 = tf.add_paragraph()
        p2.text = sdesc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_DARK

    prs.save(output_path)
    print(f"Presentacion generada 100% fiel al template: {output_path} ({os.path.getsize(output_path):,} bytes, {len(prs.slides)} slides)")

if __name__ == "__main__":
    build_presentation_from_template()
