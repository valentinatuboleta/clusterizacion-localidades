import os
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_full_presentation(
    template_path="template.pptx",
    output_path="Presentacion_EDA_Clusterizacion_Localidades.pptx"
):
    prs = Presentation(template_path)
    
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
    COLOR_MUTED = RGBColor(108, 117, 125)
    
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
                p.text = "Analisis Exploratorio de Datos (EDA)\nDescomposicion Paso a Paso de Localidades"
                p.font.name = FONT_NAME
                p.font.size = Pt(26)
                p.font.bold = True
                p.font.color.rgb = COLOR_NAVY
                
                shape.left = Inches(1.2)
                shape.top = Inches(2.2)
                shape.width = Inches(8.8)
                shape.height = Inches(1.5)
            elif "Equipo BI" in shape.text:
                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = "Pipeline NLP + Metricas Relativas + Espacio Mixto | Equipo BI & Data Science TuBoleta"
                p.font.name = FONT_NAME
                p.font.size = Pt(13)
                p.font.bold = False
                p.font.color.rgb = COLOR_STEEL
                
                shape.left = Inches(1.2)
                shape.top = Inches(3.8)
                shape.width = Inches(8.8)
                shape.height = Inches(0.6)

    def create_slide_base(tema_text, desc_text):
        new_slide = prs.slides.add_slide(template_generic.slide_layout)
        
        img_part = template_generic.part.related_part("rId3")
        new_rId = new_slide.part.relate_to(
            img_part,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        
        bg_elem = copy.deepcopy(template_generic._element.xpath(".//p:bg")[0])
        blip = bg_elem.xpath(".//a:blip")[0]
        blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", new_rId)
        
        cSld = new_slide._element.xpath(".//p:cSld")[0]
        if len(cSld.xpath("./p:bg")) > 0:
            cSld.replace(cSld.xpath("./p:bg")[0], bg_elem)
        else:
            cSld.insert(0, bg_elem)
            
        for sh in list(new_slide.shapes):
            sp = sh._element
            sp.getparent().remove(sp)
            
        tb_tema = new_slide.shapes.add_textbox(Inches(0.92), Inches(0.19), Inches(8.16), Inches(0.40))
        tf_tema = tb_tema.text_frame
        tf_tema.word_wrap = True
        p_tema = tf_tema.paragraphs[0]
        p_tema.text = tema_text.upper()
        p_tema.font.name = FONT_NAME
        p_tema.font.size = Pt(18)
        p_tema.font.bold = True
        p_tema.font.color.rgb = COLOR_WHITE
        
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

    def add_chart_slide(tema, desc, fig_path, analysis_text, conclusions_list):
        slide = create_slide_base(tema, desc)
        
        # Tarjeta Izquierda (Imagen)
        card_img = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(5.9), Inches(5.0))
        card_img.fill.solid()
        card_img.fill.fore_color.rgb = COLOR_CARD_BG
        card_img.line.color.rgb = COLOR_CARD_BORDER
        slide.shapes.add_picture(fig_path, Inches(0.95), Inches(1.95), width=Inches(5.6))
        
        # Tarjeta Derecha (Texto)
        card_txt = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.85), Inches(5.6), Inches(5.0))
        card_txt.fill.solid()
        card_txt.fill.fore_color.rgb = COLOR_CARD_BG
        card_txt.line.color.rgb = COLOR_CARD_BORDER
        
        tb = slide.shapes.add_textbox(Inches(7.1), Inches(1.95), Inches(5.2), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_an_title = tf.paragraphs[0]
        p_an_title.text = "ANALISIS DE DATOS"
        p_an_title.font.name = FONT_NAME
        p_an_title.font.size = Pt(13)
        p_an_title.font.bold = True
        p_an_title.font.color.rgb = COLOR_STEEL
        p_an_title.space_after = Pt(2)
        
        p_an = tf.add_paragraph()
        p_an.text = analysis_text
        p_an.font.name = FONT_NAME
        p_an.font.size = Pt(10.5)
        p_an.font.color.rgb = COLOR_DARK
        p_an.space_after = Pt(5)
        
        p_con_title = tf.add_paragraph()
        p_con_title.text = "TRES CONCLUSIONES CLAVE"
        p_con_title.font.name = FONT_NAME
        p_con_title.font.size = Pt(13)
        p_con_title.font.bold = True
        p_con_title.font.color.rgb = COLOR_TEAL
        p_con_title.space_after = Pt(2)
        
        for c_idx, c_text in enumerate(conclusions_list):
            p_c = tf.add_paragraph()
            p_c.text = str(c_idx+1) + ". " + c_text
            p_c.font.name = FONT_NAME
            p_c.font.size = Pt(10.5)
            p_c.font.color.rgb = COLOR_DARK
            p_c.space_after = Pt(2)

    def add_cards_slide(tema, desc, cards_data, layout_type="3_cols"):
        slide = create_slide_base(tema, desc)
        
        if layout_type == "3_cols":
            lefts = [Inches(0.8), Inches(4.8), Inches(8.8)]
            width = Inches(3.7)
            height = Inches(4.85)
            for i, (title_c, desc_c) in enumerate(cards_data):
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.85), width, height)
                card.fill.solid()
                card.fill.fore_color.rgb = COLOR_CARD_BG
                card.line.color.rgb = COLOR_CARD_BORDER
                
                tb = slide.shapes.add_textbox(lefts[i] + Inches(0.2), Inches(2.05), width - Inches(0.4), height - Inches(0.4))
                tf = tb.text_frame
                tf.word_wrap = True
                
                p1 = tf.paragraphs[0]
                p1.text = title_c
                p1.font.name = FONT_NAME
                p1.font.size = Pt(14)
                p1.font.bold = True
                p1.font.color.rgb = COLOR_NAVY
                p1.space_after = Pt(10)
                
                p2 = tf.add_paragraph()
                p2.text = desc_c
                p2.font.name = FONT_NAME
                p2.font.size = Pt(12)
                p2.font.color.rgb = COLOR_DARK
        elif layout_type == "3_rows":
            y_pos = [Inches(1.85), Inches(3.55), Inches(5.25)]
            for i, (title_c, desc_c) in enumerate(cards_data):
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos[i], Inches(11.7), Inches(1.5))
                card.fill.solid()
                card.fill.fore_color.rgb = COLOR_CARD_BG
                card.line.color.rgb = COLOR_CARD_BORDER
                
                tb = slide.shapes.add_textbox(Inches(1.0), y_pos[i] + Inches(0.1), Inches(11.3), Inches(1.3))
                tf = tb.text_frame
                tf.word_wrap = True
                
                p1 = tf.paragraphs[0]
                p1.text = title_c
                p1.font.name = FONT_NAME
                p1.font.size = Pt(14)
                p1.font.bold = True
                p1.font.color.rgb = COLOR_STEEL
                p1.space_after = Pt(3)
                
                p2 = tf.add_paragraph()
                p2.text = desc_c
                p2.font.name = FONT_NAME
                p2.font.size = Pt(11.5)
                p2.font.color.rgb = COLOR_DARK
        elif layout_type == "4_rows":
            y_pos = [Inches(1.85), Inches(3.1), Inches(4.35), Inches(5.6)]
            for i, (title_c, desc_c) in enumerate(cards_data):
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos[i], Inches(11.7), Inches(1.1))
                card.fill.solid()
                card.fill.fore_color.rgb = COLOR_CARD_BG
                card.line.color.rgb = COLOR_CARD_BORDER
                
                tb = slide.shapes.add_textbox(Inches(1.0), y_pos[i] + Inches(0.08), Inches(11.3), Inches(0.95))
                tf = tb.text_frame
                tf.word_wrap = True
                
                p1 = tf.paragraphs[0]
                p1.text = title_c
                p1.font.name = FONT_NAME
                p1.font.size = Pt(13.5)
                p1.font.bold = True
                p1.font.color.rgb = COLOR_NAVY
                p1.space_after = Pt(2)
                
                p2 = tf.add_paragraph()
                p2.text = desc_c
                p2.font.name = FONT_NAME
                p2.font.size = Pt(11.5)
                p2.font.color.rgb = COLOR_DARK
        elif layout_type == "2_cols":
            lefts = [Inches(0.8), Inches(6.8)]
            width = Inches(5.7)
            height = Inches(4.85)
            for i, (title_c, desc_c) in enumerate(cards_data):
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lefts[i], Inches(1.85), width, height)
                card.fill.solid()
                card.fill.fore_color.rgb = COLOR_CARD_BG
                card.line.color.rgb = COLOR_CARD_BORDER
                
                tb = slide.shapes.add_textbox(lefts[i] + Inches(0.25), Inches(2.05), width - Inches(0.5), height - Inches(0.4))
                tf = tb.text_frame
                tf.word_wrap = True
                
                p1 = tf.paragraphs[0]
                p1.text = title_c
                p1.font.name = FONT_NAME
                p1.font.size = Pt(14)
                p1.font.bold = True
                p1.font.color.rgb = COLOR_STEEL
                p1.space_after = Pt(8)
                
                p2 = tf.add_paragraph()
                p2.text = desc_c
                p2.font.name = FONT_NAME
                p2.font.size = Pt(11.5)
                p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # 2. SLIDE 2 (Celdas 1-2): Configuración del Entorno y Modularización
    # -------------------------------------------------------------
    slide2 = template_generic
    for sh in list(slide2.shapes):
        sp = sh._element
        sp.getparent().remove(sp)
    tb_tema2 = slide2.shapes.add_textbox(Inches(0.92), Inches(0.19), Inches(8.16), Inches(0.40))
    p_t2 = tb_tema2.text_frame.paragraphs[0]
    p_t2.text = "1. CONFIGURACION Y MODULARIZACION"
    p_t2.font.name = FONT_NAME
    p_t2.font.size = Pt(18)
    p_t2.font.bold = True
    p_t2.font.color.rgb = COLOR_WHITE
    tb_desc2 = slide2.shapes.add_textbox(Inches(0.45), Inches(1.30), Inches(11.62), Inches(0.40))
    p_d2 = tb_desc2.text_frame.paragraphs[0]
    p_d2.alignment = PP_ALIGN.CENTER
    p_d2.text = "Estructura Desacoplada del Entorno en src/ y Reproducibilidad"
    p_d2.font.name = FONT_NAME
    p_d2.font.size = Pt(14)
    p_d2.font.bold = True
    p_d2.font.color.rgb = COLOR_STEEL

    cards_s2 = [
        ("Modulo 1: NLP y Semantica (src/nlp_utils.py)",
         "- normalizar_texto: Limpieza regex y unidecode.\n- limpiar_ruido_marketing: Supresion de stopwords de gira y sponsors.\n- extraer_atributos_estructurales: Deteccion de 17 tags binarios.\n- vectorizar_texto_limpio: Vectorizacion TF-IDF compacta."),
        ("Modulo 2: Feature Engineering (src/feature_engineering.py)",
         "- filtrar_consistencia_localidades: Remocion de cuotas en cero y precios invalidos.\n- calcular_metricas_relativas: Normalizacion por evento de precios y aforos.\n- preparar_dataset_enriquecido: Pipeline maestro."),
        ("Modulo 3: Clustering y Evaluacion (src/clustering.py)",
         "- construir_espacio_vectorial_mixto: RobustScaler + TF-IDF en R^37.\n- evaluar_k_optimo: Metricas de Silueta, Inercia y Davies-Bouldin.\n- entrenar_kmeans_produccion y etiquetado.")
    ]
    lefts = [Inches(0.8), Inches(4.8), Inches(8.8)]
    for i, (col_title, col_desc) in enumerate(cards_s2):
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
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_NAVY
        p1.space_after = Pt(10)
        p2 = tf.add_paragraph()
        p2.text = col_desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # 3. SLIDE 3 (Celdas 3-4): Carga y Exploración Inicial del Dataset Raw
    # -------------------------------------------------------------
    add_cards_slide(
        "2. EXPLORACION INICIAL DEL DATASET RAW",
        "Carga y Diagnostico de Integridad sobre localidades_eda.parquet",
        [
            ("1. Dimensiones y Cobertura",
             "El dataset raw contiene 33,878 filas y 16 columnas analiticas que representan la totalidad del historial de boleteria.\n\nCada registro equivale a una localidad unica asociada a una funcion (performance) y evento."),
            ("2. Atributos Clave Disponibles",
             "- Identificadores: event_id, performance_id, site (venue).\n- Texto: logical_seat_category (nombre comercial).\n- Economicos: med_unit_amt_itx (precio en COP).\n- Fisicos: dn_quota (aforo de la localidad).\n- Ventas: tickets_pago, tickets_cero, tasa_ocupacion."),
            ("3. Diagnostico de Calidad de Datos",
             "- No hay nulos en variables economicas y de aforo.\n- Se detectan 103 filas con dn_quota = 0 o inconsistencias que deben filtrarse en el modulo de consistencia.\n- Mas de 2,400 cadenas de texto comerciales unicas.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 4. SLIDE 4 (Celdas 5-6): Normalización de Texto (Paso 3.1)
    # -------------------------------------------------------------
    add_cards_slide(
        "3. MODULO NLP - PASO 3.1: NORMALIZACION",
        "Estandarizacion de Formato y Eliminacion de Caracteres Especiales",
        [
            ("Objetivo de la Funcion normalizar_texto",
             "Elimina las diferencias ortograficas y de formato que fragmentan artificialmente localidades equivalentes:\n\n1. Conversion a minusculas completas.\n2. Eliminacion de tildes y diacriticos (ej. 'Platea Mayor' -> 'platea mayor').\n3. Supresion de simbolos, guiones y caracteres no alfanumericos.\n4. Normalizacion de espacios en blanco multiples."),
            ("Evidencia en Casos Reales del Catalogo",
             "- 'PALCO OCCIDENTAL VIP!' -> 'palco occidental vip'\n- 'Grada Oriental - 2do Piso' -> 'grada oriental 2do piso'\n- 'GENERAL (Preventa #1)' -> 'general preventa 1'\n- 'VIP EARLY ACCESS / BANCO' -> 'vip early access banco'\n\nResultado: Se consolida la coherencia semantica base previa a la limpieza profunda.")
        ],
        layout_type="2_cols"
    )

    # -------------------------------------------------------------
    # 5. SLIDE 5 (Celdas 7-8): Limpieza de Ruido de Marketing (Paso 3.2)
    # -------------------------------------------------------------
    add_cards_slide(
        "3. MODULO NLP - PASO 3.2: RUIDO DE MARKETING",
        "Supresion de Nombres de Giras, Sponsors y Promociones Publicitarias",
        [
            ("1. Configuracion de Stopwords de Marketing",
             "Se definieron mas de 50 stopwords publicitarias organizadas en 4 categorias:\n- Sponsors: movistar, etb, visa, mastercard, davivienda, bancolombia, aguila, pilsen.\n- Giras: cantinero, tour, world, leyenda, ferxxo, mor, bichota, corazon.\n- Canales: preventa, etapa, early, presale, fase, combo."),
            ("2. Mecanismo Regex Limpio",
             "La funcion limpiar_ruido_marketing(texto) aplica expresiones regulares con limites de palabra (\b) para suprimir terminos de marketing sin romper palabras estructurales compuestas (ej. 'movistar arena' conserva 'arena')."),
            ("3. Impacto en la Homogeneizacion",
             "Ejemplos reales transformados:\n- 'palco cantinero movistar' -> 'palco'\n- 'modo leyenda vip presale' -> 'vip'\n- 'etb preferencial combo 2' -> 'preferencial 2'\n- 'experiencia platino aguila' -> 'platino'")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 6. SLIDE 6 (Celdas 9-10): Casos Extremos y Nombres Numéricos
    # -------------------------------------------------------------
    add_cards_slide(
        "3. MODULO NLP - PASO 3.2: CASOS EXTREMOS",
        "Manejo de Cadenas Vacias y Preservacion de Jerarquias Numericas",
        [
            ("Analisis de Cadenas Vacias Post-Limpieza (0.58%)",
             "Solo 196 registros (0.58% del catalogo) quedan vacios tras remover marketing publicitario (ej. 'Gira Nacional 2024' o 'Preventa Exclusiva').\n\nEstrategia de Mitigacion: El pipeline les asigna automaticamente la etiqueta semantica 'general' y delega su jerarquia a las metricas relativas de precio y aforo."),
            ("Preservacion de Digitos y Niveles de Fila (20.5%)",
             "6,950 localidades (20.5% del catalogo) contienen numeros vitales para la jerarquia fisica (ej. 'piso 1', 'fila a', 'tribuna 2').\n\nDecision de Arquitectura: La regex conserva los digitos para no perder la diferencia critica entre 'platea 1' (frontal) y 'platea 3' (posterior).")
        ],
        layout_type="2_cols"
    )

    # -------------------------------------------------------------
    # 7. SLIDE 7 (Celdas 11-12): Extracción de Atributos Estructurales
    # -------------------------------------------------------------
    add_cards_slide(
        "3. MODULO NLP - PASO 3.3: ATRIBUTOS ESTRUCTURALES",
        "Pipeline Regex/NER para 17 Variables Binarias en 4 Dimensiones",
        [
            ("1. Jerarquia Comercial (5 Tags)", "tag_palco, tag_vip, tag_platea, tag_preferencial, tag_general.\nIdentifican la categoria de servicio y el estatus economico percibido."),
            ("2. Nivel Vertical y Arquitectura (3 Tags)", "tag_balcon, tag_piso_alto, tag_piso_bajo.\nCapturan la estratificacion vertical en teatros de varios niveles, coliseos y arenas."),
            ("3. Orientacion Espacial (6 Tags)", "tag_occidental, tag_oriental, tag_norte, tag_sur, tag_lateral, tag_vista_parcial.\nUbican cardinalmente el asiento en estadios y festivales."),
            ("4. Restricciones de Acceso (3 Tags)", "tag_familiar, tag_menores, tag_movilidad_reducida.\nCapturan politicas especiales de ingreso y normativas de accesibilidad.")
        ],
        layout_type="4_rows"
    )

    # -------------------------------------------------------------
    # 8. SLIDE 8 (Celdas 13-14): Gráfico 1 - Frecuencia de Atributos
    # -------------------------------------------------------------
    add_chart_slide(
        "MODULO 1: DESCOMPOSICION NLP",
        "Grafico 1: Frecuencia de Atributos Estructurales, Espaciales y Restricciones Extraidos",
        "reports/figures/fig1_tags_frecuencia.png",
        "De las 33,775 localidades evaluadas, la etiqueta mas frecuente es GENERAL con 15,383 apariciones (45.5%), seguida de niveles de teatro y recintos cerrados como PLATEA (4,015), PALCO (3,547), PISO_ALTO (2,933) y BALCON (2,679). En el ambito espacial, las orientaciones predominantes son LATERAL (1,377) y OCCIDENTAL (1,134).",
        [
            "Predominio de Localidades Masivas: Casi la mitad del inventario corresponde a admision general, lo que exige diferenciar una general de estadio frente a una de teatro.",
            "Alta Especializacion en Teatros y Arenas: Mas de 14,400 registros corresponden a balcones, plateas, pisos altos y palcos con distribucion vertical escalonada.",
            "Efectividad del Pipeline Regex/NER: La deteccion de 17 etiquetas permitio convertir texto libre en variables numericas estructuradas de jerarquia, orientacion y restricciones."
        ]
    )

    # -------------------------------------------------------------
    # 9. SLIDE 9 (Celdas 15-16): Vectorización TF-IDF Estructurada (Paso 3.4)
    # -------------------------------------------------------------
    add_cards_slide(
        "3. MODULO NLP - PASO 3.4: VECTORIZACION TF-IDF",
        "Representacion N-Gram Compacta sobre Texto Estructurado",
        [
            ("1. Configuracion del Vectorizador",
             "Se ajusta un TfidfVectorizer(max_features=15, ngram_range=(1,2)) entrenado exclusivamente sobre la columna logical_category_clean.\n\nAl haber eliminado previamente el ruido publicitario, el vocabulario captura senales semanticas puras."),
            ("2. Vocabulario Compacto de 15 Tokens",
             "Tokens aprendidos por la matriz TF-IDF:\n'alto', 'balcon', 'general', 'lateral', 'norte', 'occidental', 'oriental', 'palco', 'piso', 'piso alto', 'platea', 'preferencial', 'sur', 'vip', 'vista'.\n\nRepresentan el 99.1% de la masa semantica del catalogo."),
            ("3. Integracion en el Espacio Vectorial",
             "Aporta 15 dimensiones continuas que complementan las 17 variables binarias, capturando intensidades de texto y combinaciones n-gramicas sin sobredimensionar la matriz final.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 10. SLIDE 10 (Celdas 17-18): Filtrado de Consistencia (Paso 4.1)
    # -------------------------------------------------------------
    add_cards_slide(
        "4. FEATURE ENGINEERING - PASO 4.1: FILTRADO",
        "Reglas de Depuracion y Consistencia Fisica de Localidades",
        [
            ("Reglas de Filtrado Aplicadas (filtrar_consistencia_localidades)",
             "1. Exclusion de Cuotas en Cero: Se descartan registros con dn_quota <= 0 (localidades de prueba tecnica o bloqueadas administrativamente).\n2. Exclusion de Precios Invalidos: Se descartan boletas con med_unit_amt_itx < 0.\n3. Eliminacion de Eventos Inconsistentes: Funciones donde la suma de aforos de localidades sea cero.\n\nResultado: Se eliminan 103 registros corruptos (0.3% del dataset), consolidando un universo depurado de 33,775 localidades."),
            ("Impacto en la Estabilidad de Ratios",
             "- Garantiza que el denominador en peso_aforo (performance_quota) nunca genere divisiones por cero o infinitos.\n- Permite calcular percentiles de precio y ratios relativos exactos y reproducibles en produccion.")
        ],
        layout_type="2_cols"
    )

    # -------------------------------------------------------------
    # 11. SLIDE 11 (Celdas 19-20): Cálculo de Métricas Relativas (Paso 4.2)
    # -------------------------------------------------------------
    add_cards_slide(
        "4. FEATURE ENGINEERING - PASO 4.2: METRICAS RELATIVAS",
        "Formulacion Matematica de Variables Normalizadas por Evento",
        [
            ("ratio_precio_max (Cardinal Económico)",
             "P / max(P_evento)\nMide que porcentaje representa el precio de la localidad frente a la localidad mas cara de esa misma funcion. Rango [0.0, 1.0]."),
            ("percentil_precio_evento (Ordinal de Jerarquía)",
             "rank(P_evento) / K_localidades\nMide el escalon jerarquico de la boleta dentro del show, independiente de si la brecha en dinero es de $10k o $1M COP."),
            ("peso_aforo (Escala de Capacidad Física)",
             "dn_quota / performance_quota\nCaptura la fraccion fisica del recinto que ocupa la localidad (ej. Palco 5% vs Grada 40%)."),
            ("tasa_ocupacion y tasa_venta_paga",
             "tickets_vendidos / dn_quota\nMiden la velocidad de absorcion y demanda historica del publico por localidad.")
        ],
        layout_type="4_rows"
    )

    # -------------------------------------------------------------
    # 12. SLIDE 12 (Celdas 21-22): Gráfico 2 - Distribuciones Catálogo Total
    # -------------------------------------------------------------
    add_chart_slide(
        "5. ANALISIS DE DISTRIBUCIONES",
        "Grafico 2: Distribucion de Variables Relativas Normalizadas (Catalogo Total, N = 33,775)",
        "reports/figures/fig2_distribuciones.png",
        "El ratio_precio_max presenta una concentracion alta en 1.0 (mediana = 1.0, media = 0.803). El peso_aforo muestra una clara bimodalidad: una concentracion de zonas exclusivas de aforo reducido (percentil 25 = 0.049, menos del 5% del venue) frente a eventos de admision unica. La tasa_ocupacion media es de 18.1%.",
        [
            "Desacople Exitoso de la Moneda: La escala 0.0 a 1.0 permite comparar precios de festivales masivos con teatros locales bajo la misma regla de exclusividad.",
            "Identificacion Inmediata de Zonas Selectas: El 25% de las localidades ocupan menos del 5% del aforo total, constituyendo candidatas naturales a VIP o palcos.",
            "Descomposicion del Pico en 1.0: El 45.5% corresponde a eventos de admision unica (tarifa plana) y el 13.0% a las localidades mas caras en eventos multi-zona."
        ]
    )

    # -------------------------------------------------------------
    # 13. SLIDE 13 (Celdas 23-24): Descomposición de ratio_precio_max = 1.0
    # -------------------------------------------------------------
    add_cards_slide(
        "5. DESCOMPOSICION DEL RATIO DE PRECIO = 1.0",
        "Demostracion Matematica: Cardinalidad de Eventos y Desglose de Tarifas Planas",
        [
            ("1. Eventos de Admision Unica (filas_peso_1 = 15,375 | 45.5%)",
             "Funciones con una sola localidad donde peso_aforo = 100% (tarifa plana).\nPor definicion matematica, el unico precio existente es el maximo (ratio_precio_max = 1.0)."),
            ("2. Localidades Top Multi-Zona (filas_ratio_1_multizona = 4,407 | 13.0%)",
             "La localidad o grupo de localidades mas costosas dentro de espectaculos estratificados (obras de teatro con platea/balcon, estadios con palcos/gradas, etc.)."),
            ("3. Cardinalidad y Union Disjunta (filas_ratio_1 = 19,782 | 58.6%)",
             "El 82.5% de los eventos del catalogo tienen 1 sola localidad. En eventos multi-zona (17.5%), el promedio es de 5.7 localidades por funcion.\n\nfilas_peso_1 subconjunto estricto de filas_ratio_1.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 14. SLIDE 14 (Celdas 25-26): Top Venues de Admisión Única
    # -------------------------------------------------------------
    add_chart_slide(
        "HALLAZGOS EMPIRICOS: ADMISION UNICA",
        "Top Venues y Caracteristicas de Eventos de Localidad Unica (N = 15,375)",
        "reports/figures/fig2b_top_venues_unica.png",
        "Los eventos con una sola localidad (peso_aforo = 100%) representan el 82.5% de todos los eventos del catalogo. Los datos demuestran que estan concentrados masivamente en funciones culturales y ciclos continuos de cine, centros interactivos de ciencia, planetarios y comedia en vivo.",
        [
            "Concentracion en Salas de Cine: Mas de 7,000 funciones (45.6% del total) se realizan en las salas de la Cinemateca de Bogota (Sala 3: 2,392, Sala Capital: 2,379, Sala 2: 2,235).",
            "Centros de Ciencia y Museos: Recintos interactivos como YAWA Cali (1,476 funciones), Maloka (864), Planetario (421) y Museo La Tertulia (303) operan con tarifa plana.",
            "Comedia y Teatros de Camara: Espacios como Boom Stand Up Bar (810) y Teatro Petra (471) manejan acceso general no zonificado."
        ]
    )

    # -------------------------------------------------------------
    # 15. SLIDE 15 (Celdas 27-28): Demostración Multi-Zona: Palcos y VIPs
    # -------------------------------------------------------------
    add_chart_slide(
        "DEMOSTRACION EMPIRICA: MULTI-ZONA",
        "¿Son Realmente Palcos y VIPs las Localidades Mas Caras en Eventos Multi-Zona? (N = 4,407)",
        "reports/figures/fig2c_multizona_tags_top.png",
        "Al evaluar las 4,407 localidades con ratio_precio_max = 1.0 en eventos multi-zona (peso_aforo < 0.99), el 61.5% activa tags explicitos de gama alta: PLATEA (1,450 reg | 32.9%), PALCO (842 reg | 19.1%), PREFERENCIAL (474 reg | 10.8%) y VIP (401 reg | 9.1%).",
        [
            "Tipologia por Tipo de Recinto: En teatros y auditorios, la localidad mas costosa es Platea / Luneta; en estadios y arenas son Palcos / VIPs; en festivales son Zonas Fan / Coche.",
            "Dominio de Etiquetas de Lujo: Las 4 etiquetas de mayor precio suman 3,167 registros de las 4,407 localidades tope.",
            "Nombres Creativos Clasificados por Precio: Localidades con nombres de fantasia sin palabra VIP se clasifican por su ratio = 1.0 y bajo aforo."
        ]
    )

    # -------------------------------------------------------------
    # 16. SLIDE 16 (Celdas 29-30): Gráfico 2B - Distribuciones Multi-Zona
    # -------------------------------------------------------------
    add_chart_slide(
        "DISTRIBUCIONES EN EVENTOS MULTI-ZONA",
        "Grafico 2B: Descompresion de Variables Relativas sin Admision Unica (N = 18,400)",
        "reports/figures/fig2d_distribuciones_multizona.png",
        "Al aislar las 18,400 localidades en 3,252 eventos multi-zona (peso_aforo < 0.99), se elimina el sesgo del 45.5% de tarifas planas. La mediana de ratio_precio_max se reduce de 1.00 a 0.671 y la mediana de peso_aforo baja de 1.00 a 0.111 (11.1% del recinto).",
        [
            "Descompresion Real de Precios: Emerge una distribucion continua donde gradas populares estan en 0.20-0.50, preferenciales en 0.60-0.85 y VIPs en 1.00.",
            "Capacidad Fisica Realista: El 75% de las localidades multi-zona ocupan menos del 24.5% del aforo del recinto, reflejando la arquitectura real de arenas y estadios.",
            "Simetria y Equilibrio del Percentil: El percentil relativo se convierte en una curva simetrica y balanceada (mediana 0.600) ideal para optimizacion en K-Means."
        ]
    )

    # -------------------------------------------------------------
    # 17. SLIDE 17 (Celdas 31-32): Evaluación Cuantitativa de las 17 Variables
    # -------------------------------------------------------------
    add_cards_slide(
        "EVALUACION DE LAS 17 VARIABLES NLP",
        "Estadisticas Descriptivas de las 4 Dimensiones Ortogonales en el Catalogo",
        [
            ("1. Jerarquia Comercial (5 Tags)",
             "- GENERAL (15,383 reg | 45.5%): Ratio medio 0.96, Aforo 95.3%, Mediana $15k COP.\n- PLATEA (4,015 reg | 11.9%): Ratio medio 0.80, Aforo 22.7%, Mediana $95k COP.\n- PALCO (3,547 reg | 10.5%): Ratio medio 0.69, Aforo 9.8%, Mediana $108k COP.\n- VIP (1,253 reg | 3.7%): Ratio medio 0.68, Aforo 15.9%, Mediana $190k COP.\n- PREFERENCIAL (1,253 reg | 3.7%): Ratio medio 0.80, Aforo 19.0%, Mediana $73.5k COP."),
            ("2. Nivel Vertical y Espacial (9 Tags)",
             "- PISO_ALTO (2,933 reg | 8.7%): Ratio medio 0.46 (gradas altas economicas), Mediana $96.5k COP.\n- BALCON (2,679 reg | 7.9%): Ratio medio 0.58, Aforo 16.2%, Mediana $56k COP.\n- OCCIDENTAL (1,134 reg): Ratio medio 0.51, Desv 0.35, Mediana $50k COP.\n- ORIENTAL (799 reg): Ratio medio 0.52, Desv 0.26, Mediana $44k COP."),
            ("3. Restricciones de Acceso (3 Tags)",
             "- FAMILIAR (82 reg | 0.2%): Ratio medio 0.46, Aforo 9.7%.\n- MOVILIDAD_REDUCIDA (63 reg | 0.2%): Ratio medio 0.77, Aforo 0.7%.\n- MENORES (57 reg | 0.2%): Ratio medio 0.45, Aforo 12.4%.\n\nSuman menos del 0.3% del catalogo.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 18. SLIDE 18 (Celdas 32-33): Demostración Visual 17 Tags
    # -------------------------------------------------------------
    add_chart_slide(
        "VALIDACION DE VARIABLES NLP",
        "Demostracion Visual: Volumen Muestral y Capacidad Discriminante (Precio vs Aforo)",
        "reports/figures/fig3a_demostracion_17tags.png",
        "El scatter plot bivariado demuestra como los tags de Jerarquia Comercial y Nivel Vertical forman una curva diagonal de segmentacion perfecta (desde General con 95% aforo hasta VIP con $190k COP), mientras que todas las etiquetas de Orientacion Espacial (Occidental, Oriental, Norte, Sur) colapsan en el mismo racimo central (~$40k-$58k COP y ~10%-19% aforo).",
        [
            "Soporte Muestral Solido (>95%): Los 7 tags seleccionados concentran mas del 95% de las asignaciones de etiquetas del catalogo.",
            "Ortogonalidad de la Orientacion Espacial: Occidental tiene un rango de precios desde $0 hasta $2.15M COP (mezcla 113 VIPs, 202 Gradas Altas y 181 Generales).",
            "Bajo Volumen en Restricciones: Familiar, Menores y Movilidad Reducida representan menos del 0.3% y atienden a normas legales de acceso, no a pricing."
        ]
    )

    # -------------------------------------------------------------
    # 19. SLIDE 19 (Celdas 33-34): Justificación de los 7 Tags Principales
    # -------------------------------------------------------------
    add_cards_slide(
        "SELECCION DE LOS 7 TAGS PRINCIPALES",
        "Justificacion Matematica y de Negocio para la Matriz de Clustering",
        [
            ("1. Poder Discriminante Monotono",
             "Las 5 etiquetas de Jerarquia Comercial (PALCO, VIP, PLATEA, PREFERENCIAL, GENERAL) junto a 2 de Nivel Vertical (BALCON, PISO_ALTO) presentan una separacion lineal limpia en el espacio euclidiano:\n- GENERAL: Base accesible y aforo masivo.\n- BALCON / PISO ALTO: Localidades de altura y precio moderado.\n- PLATEA / PREFERENCIAL: Gama media-alta frontal.\n- PALCO / VIP: Maxima exclusividad y precio tope."),
            ("2. Ortogonalidad de Orientaciones Espaciales",
             "Incluir Occidental u Oriental como variables discriminantes de precio anadiria ruido al algoritmo, ya que una localidad 'Occidental' puede ser tanto la mas costosa (Palco Occidental) como la mas economica (Occidental Alta General).\n\nSu rol es de orientacion geografica complementaria."),
            ("3. Consolidacion de los 4 Arquetipos",
             "Estos 7 tags constituyen los pilares que alimentan directamente los 4 Arquetipos de Demanda en la clusterizacion K-Means.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 20. SLIDE 20 (Celdas 35-36): Gráfico 3 - Boxplots Bivariados
    # -------------------------------------------------------------
    add_chart_slide(
        "VALIDACION SEMANTICA Y FISICA",
        "Grafico 3: Comparacion Bivariada de Precio Relativo y Aforo por Atributo NLP",
        "reports/figures/fig3b_boxplots_bivariados.png",
        "PALCO y VIP registran las medianas de peso_aforo mas reducidas del catalogo (9.8% y 15.9% del venue), manteniendo ratios de precio con maximos en 1.0 y precio mediano de $108k y $190k. PREFERENCIAL y PLATEA registran ratios de precio de 0.80 con aforos medios (19% a 23%). GENERAL concentra el 95.3% de aforo y $15k de precio.",
        [
            "Confirmacion de Jerarquia Fisica: La semantica NLP se alinea con la capacidad fisica: los palcos ocupan fracciones minimas de aforo y las generales absorben el volumen.",
            "Validacion de la Platea como Zona Preferente: Las plateas se ubican en el rango superior de precios ($95k), consolidandose como el escalon intermedio-alto.",
            "El Balcon como Opcion Accesible de Recinto Cerrado: Los balcones registran sistematicamente precios moderados ($56k), validando su rol accesible en teatro."
        ]
    )

    # -------------------------------------------------------------
    # 21. SLIDE 21 (Celda 37): Gráfico 4 - Matriz de Correlaciones
    # -------------------------------------------------------------
    add_chart_slide(
        "ANALISIS DE CORRELACIONES",
        "Grafico 4: Matriz de Correlaciones Numericas y Ratios Estructurales",
        "reports/figures/fig4_correlaciones.png",
        "La matriz de Pearson evalua relaciones lineales entre dinero nominal en COP (med_unit_amt_itx), ratios de precio (ratio_precio_max, ratio_precio_mean, percentil_precio_evento), capacidad fisica (dn_quota, peso_aforo) y rendimiento (tasa_ocupacion, tasa_venta_paga, ratio_cortesias).",
        [
            "Correlacion Nula con el Dinero Nominal: r(med_unit_amt_itx, ratio_precio_max) = -0.034. El dinero en pesos COP es independiente de la exclusividad.",
            "Relacion Negativa de Aforo y Precio: r(peso_aforo, med_unit_amt_itx) = -0.247 y r(peso_aforo, tasa_ocupacion) = -0.331.",
            "Ausencia de Colinealidad Critica: Todas las variables presentan correlaciones |r| < 0.85, garantizando complementariedad en el espacio vectorial."
        ]
    )

    # -------------------------------------------------------------
    # 22. SLIDE 22 (Celda 38): Análisis Detallado de Correlaciones
    # -------------------------------------------------------------
    add_cards_slide(
        "DETALLE DE CONCLUSIONES: CORRELACIONES",
        "Tres Principios Econometricos que Respaldan el Modelo de Clustering",
        [
            ("1. Independencia del Dinero en COP (r = -0.034)",
             "- Falacia del Valor Absoluto: Una boleta de $150k COP es General en concierto de estadio de Karol G (percentil 0.15), pero Platea VIP en obra de teatro (percentil 1.00).\n- Beneficio Machine Learning: Las metricas relativas vuelven al modelo invariante a la inflacion, al tipo de show y al tamano del venue."),
            ("2. Ley de Oferta y Demanda en el Aforo (r = -0.247)",
             "- Principio de Escasez: Palcos y VIPs ocupan entre 1% y 10% del venue, permitiendo fijar precios maximos.\n- Economias de Escala Masiva: Gradas generales ocupan 30% a 60% del venue y requieren precios populares para llenar masa critica."),
            ("3. No Redundancia y Distancia Euclidiana (|r| < 0.85)",
             "- ratio_precio_max vs percentil_precio (r = 0.848): ratio aporta magnitud cardinal (%) y percentil aporta orden jerarquico ordinal.\n- Previene sesgos por multicolinealidad en la distancia euclidiana de K-Means.")
        ],
        layout_type="3_cols"
    )

    # -------------------------------------------------------------
    # 23. SLIDE 23: Gráfico 5 y 4 Arquetipos Universales de Demanda
    # -------------------------------------------------------------
    slide23 = create_slide_base(
        "ESPACIO DE CLUSTERING Y ARQUETIPOS",
        "Grafico 5: Cuadrantes de Demanda y Estandarizacion en 4 Segmentos Universales"
    )
    
    # Tarjeta Izquierda (Imagen Gráfico 5)
    card_img = slide23.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0))
    card_img.fill.solid()
    card_img.fill.fore_color.rgb = COLOR_CARD_BG
    card_img.line.color.rgb = COLOR_CARD_BORDER
    slide23.shapes.add_picture("reports/figures/fig5_cuadrantes_demanda.png", Inches(0.95), Inches(1.95), width=Inches(5.3))
    
    # Tarjeta Derecha (4 Arquetipos)
    card_txt = slide23.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.85), Inches(5.9), Inches(5.0))
    card_txt.fill.solid()
    card_txt.fill.fore_color.rgb = COLOR_CARD_BG
    card_txt.line.color.rgb = COLOR_CARD_BORDER
    
    tb23 = slide23.shapes.add_textbox(Inches(6.8), Inches(1.95), Inches(5.5), Inches(4.8))
    tf23 = tb23.text_frame
    tf23.word_wrap = True
    
    p_t = tf23.paragraphs[0]
    p_t.text = "4 ARQUETIPOS UNIVERSALES DE DEMANDA"
    p_t.font.name = FONT_NAME
    p_t.font.size = Pt(13)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_NAVY
    p_t.space_after = Pt(4)
    
    archetypes_summary = [
        ("1. VIP / Palcos / Premium (1,919 | 5.7%)", "Ratio: 0.72 | Aforo: 19.1% | Ocupacion: 69.7%\nPalcos, Suites, Boxes, Experiencia Platino"),
        ("2. Preferencial / Platea Frontal (5,759 | 17.0%)", "Ratio: 0.74 | Aforo: 22.2% | Ocupacion: 40.9%\nPlatea Delantera, Preferencial 1, Sillas Centrales"),
        ("3. Grada General / Masiva (17,079 | 50.6%)", "Ratio: 0.99 | Aforo: 89.3% | Ocupacion: 9.1%\nGeneral, Entrada Unica, Tiquete Full, Admision General"),
        ("4. Popular / Balcon / Vista Parcial (9,018 | 26.7%)", "Ratio: 0.49 | Aforo: 18.3% | Ocupacion: 10.0%\nPlatea Posterior, Balcon Mayor, 2do Piso, Grada Alta")
    ]
    for aname, adesc in archetypes_summary:
        pa = tf23.add_paragraph()
        pa.text = aname
        pa.font.name = FONT_NAME
        pa.font.size = Pt(11)
        pa.font.bold = True
        pa.font.color.rgb = COLOR_STEEL
        pa.space_after = Pt(1)
        
        pd_ = tf23.add_paragraph()
        pd_.text = adesc
        pd_.font.name = FONT_NAME
        pd_.font.size = Pt(10)
        pd_.font.color.rgb = COLOR_DARK
        pd_.space_after = Pt(3)

    prs.save(output_path)
    print(f"Presentacion exhaustiva de 23 slides generada con exito: {output_path} ({os.path.getsize(output_path):,} bytes, {len(prs.slides)} slides)")

if __name__ == "__main__":
    build_full_presentation()
