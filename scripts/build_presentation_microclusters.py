"""
Script para compilar la presentación oficial en formato PowerPoint (.pptx)
utilizando el template corporativo (template.pptx) del repositorio.

Genera:
  - Presentacion_Microclusters_Marcha_Blanca.pptx
"""

import os
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(
    template_path="template.pptx",
    output_path="Presentacion_Microclusters_Marcha_Blanca.pptx"
):
    print("==================================================================")
    print("COMPILANDO PRESENTACIÓN POWERPOINT (template.pptx)")
    print("==================================================================")

    prs = Presentation(template_path)
    
    template_cover = prs.slides[0]
    template_generic = prs.slides[1]
    
    # Paleta de colores corporativos
    COLOR_NAVY = RGBColor(11, 37, 69)       # #0B2545
    COLOR_STEEL = RGBColor(19, 64, 116)     # #134074
    COLOR_TEAL = RGBColor(0, 150, 136)      # #009688
    COLOR_CORAL = RGBColor(231, 111, 81)    # #E76F51
    COLOR_DARK = RGBColor(33, 37, 41)       # #212529
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_CARD_BORDER = RGBColor(215, 220, 228)
    COLOR_CARD_BG = RGBColor(255, 255, 255)
    COLOR_MUTED = RGBColor(108, 117, 125)
    
    FONT_NAME = "Montserrat"

    # Helper para configurar títulos y encabezados sobre cualquier slide
    def setup_slide_content(slide, tema_text, desc_text, notas=""):
        for sh in list(slide.shapes):
            sp = sh._element
            sp.getparent().remove(sp)
            
        # Encabezado de Tema
        tb_tema = slide.shapes.add_textbox(Inches(0.92), Inches(0.19), Inches(11.5), Inches(0.40))
        tf_tema = tb_tema.text_frame
        tf_tema.word_wrap = True
        p_tema = tf_tema.paragraphs[0]
        p_tema.text = tema_text.upper()
        p_tema.font.name = FONT_NAME
        p_tema.font.size = Pt(17)
        p_tema.font.bold = True
        p_tema.font.color.rgb = COLOR_WHITE
        
        # Descripción / Subtítulo
        tb_desc = slide.shapes.add_textbox(Inches(0.92), Inches(0.68), Inches(11.5), Inches(0.65))
        tf_desc = tb_desc.text_frame
        tf_desc.word_wrap = True
        p_desc = tf_desc.paragraphs[0]
        p_desc.text = desc_text
        p_desc.font.name = FONT_NAME
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = COLOR_NAVY
        
        # Notas de Orador
        if notas:
            notes_slide = slide.notes_slide
            tf_notes = notes_slide.notes_text_frame
            tf_notes.text = notas
            
        return slide

    # Helper para clonar slide genérica con fondo oficial
    def create_slide(tema_text, desc_text, notas=""):
        new_slide = prs.slides.add_slide(template_generic.slide_layout)
        
        # 1. Vincular imagen de fondo
        img_part = template_generic.part.related_part("rId3")
        new_rId = new_slide.part.relate_to(
            img_part,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        
        # 2. Copiar elemento de fondo <p:bg>
        bg_elem = copy.deepcopy(template_generic._element.xpath(".//p:bg")[0])
        blip = bg_elem.xpath(".//a:blip")[0]
        blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", new_rId)
        
        cSld = new_slide._element.xpath(".//p:cSld")[0]
        if len(cSld.xpath("./p:bg")) > 0:
            cSld.replace(cSld.xpath("./p:bg")[0], bg_elem)
        else:
            cSld.insert(0, bg_elem)
            
        return setup_slide_content(new_slide, tema_text, desc_text, notas)

    # -------------------------------------------------------------
    # SLIDE 1: PORTADA
    # -------------------------------------------------------------
    print("• Configurando Slide 1: Portada...")
    for shape in template_cover.shapes:
        if shape.has_text_frame:
            if "Titulo" in shape.text:
                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = "Micro-Clusters y Arquetipos de Demanda\nDoble Etiquetado en Marcha Blanca"
                p.font.name = FONT_NAME
                p.font.size = Pt(26)
                p.font.bold = True
                p.font.color.rgb = COLOR_NAVY
                
                shape.left = Inches(1.2)
                shape.top = Inches(2.2)
                shape.width = Inches(9.5)
                shape.height = Inches(1.5)
            elif "Equipo BI" in shape.text:
                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = "Shadow Testing en Azure Capa GOLD | Macro v2.5 + Micro v3.0 | Equipo BI & Data Science TuBoleta"
                p.font.name = FONT_NAME
                p.font.size = Pt(13)
                p.font.bold = False
                p.font.color.rgb = COLOR_STEEL
                
                shape.left = Inches(1.2)
                shape.top = Inches(4.0)
                shape.width = Inches(9.5)
                shape.height = Inches(0.6)

    template_cover.notes_slide.notes_text_frame.text = (
        "Hoy presentamos los resultados de la marcha blanca sobre datos transaccionales reales descargados de Azure. "
        "Ambos modelos corrieron congelados, sin re-entrenar una sola línea, demostrando cómo opera el pipeline en producción real."
    )

    # -------------------------------------------------------------
    # SLIDE 2: EL MODELO EN DOS NIVELES (FIGURA 1)
    # -------------------------------------------------------------
    print("• Configurando Slide 2: El Modelo en Dos Niveles...")
    s2 = setup_slide_content(
        template_generic,
        "ARQUITECTURA JERÁRQUICA DE CLASIFICACIÓN",
        "El Modelo en Dos Niveles: Del Micro-Cluster Técnico al Arquetipo de Negocio (Rollup 1:1)",
        "Fíjense en la estructura de árbol: el negocio habla en términos de 6 arquetipos para planear eventos, "
        "pero el motor de datos opera en micro-clusters. El puente entre la ingeniería y el negocio es label_auto, "
        "garantizando que todo agrupamiento técnico tenga un sentido comercial transparente."
    )
    fig1_path = "reports/figures/fig_arbol_arquetipos_microclusters.png"
    if os.path.exists(fig1_path):
        s2.shapes.add_picture(fig1_path, Inches(0.8), Inches(1.4), Inches(8.0), Inches(5.6))
        
    # Tarjeta de Bullets a la derecha
    card_s2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.0), Inches(1.4), Inches(3.5), Inches(5.6))
    card_s2.fill.solid()
    card_s2.fill.fore_color.rgb = COLOR_CARD_BG
    card_s2.line.color.rgb = COLOR_CARD_BORDER
    tb_s2 = s2.shapes.add_textbox(Inches(9.15), Inches(1.5), Inches(3.2), Inches(5.3))
    tf_s2 = tb_s2.text_frame
    tf_s2.word_wrap = True
    
    bullets_s2 = [
        ("Nivel 1 (Macro de Negocio):", "6 Arquetipos de demanda que gobiernan pricing y contingentes."),
        ("Nivel 2 (Micro-Cluster):", "Particiones especializadas por patrones léxicos y arquitectónicos."),
        ("Invarianza Rollup 1:1:", "Todo micro-cluster pertenece estrictamente a un único arquetipo."),
        ("El Puente Semántico:", "label_auto traduce centroides numéricos a etiquetas legibles de negocio.")
    ]
    for b_title, b_desc in bullets_s2:
        p1 = tf_s2.add_paragraph() if tf_s2.paragraphs[0].text else tf_s2.paragraphs[0]
        p1.text = b_title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_STEEL
        p1.space_after = Pt(2)
        
        p2 = tf_s2.add_paragraph()
        p2.text = b_desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_DARK
        p2.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 3: EL CATÁLOGO CANÓNICO (V3.0)
    # -------------------------------------------------------------
    print("• Configurando Slide 3: El Catálogo Canónico...")
    s3 = create_slide(
        "CATÁLOGO CANÓNICO DE MICRO-CLUSTERS",
        "Estructura Formal de los 18 Micro-Clusters Activos, Pureza Léxica y Estabilidad",
        "El catálogo v3.0 no fue producto de intuición: cada sub-espacio se optimizó independientemente con curvas Codo-Davies Bouldin. "
        "Excluimos deliberadamente dos grupos degenerados que tenían solo 3 filas en pruebas tempranas para mantener un catálogo industrial "
        "robusto con estabilidad bootstrap superior a 0.88 en todas las particiones clave."
    )
    # 3 Columnas de tarjetas
    cat_cols = [
        ("Admisión Única y Popular", [
            ("AU-0 | Admisión Única", "15,375 locs (45.5%) | Nodo terminal determinístico puro."),
            ("POP-0 | hibrido_k0", "1,549 locs | Balcón dominante (40.0% pureza)."),
            ("POP-1 | hibrido_k1", "1,506 locs | Piso alto dominante (41.2% pureza)."),
            ("POP-2 | hibrido_k2", "1,576 locs | Balcón y piso alto (55.1% pureza)."),
            ("POP-3 | hibrido_k3", "1,223 locs | Pisos superiores de arena/estadio.")
        ]),
        ("VIP y Preferencial Frontal", [
            ("VIP-0 | hibrido_k0", "799 locs | Palcos mixtos con balcón."),
            ("VIP-1 | hibrido_k1", "2,347 locs | Palcos teatro y suites (51.0%)."),
            ("VIP-2 | hibrido_k2", "2,029 locs | Palcos puros de alta exclusividad (58.8%)."),
            ("PPF-0 | Platea", "687 locs | Platea pura frontal (99.8% pureza)."),
            ("PPF-1 | Platea", "838 locs | Platea teatro canónica (93.4% pureza)."),
            ("PPF-2 | Platea", "1,552 locs | Platea delantera y central (98.1% pureza).")
        ]),
        ("Platea General y Grada Masiva", [
            ("PGI-0 | hibrido_k0", "768 locs | Preferencial intermedio."),
            ("PGI-1 | hibrido_k1", "840 locs | Lunetas y balcones bajos."),
            ("PGI-2 | hibrido_k2", "782 locs | Lunetas intermedias de teatro."),
            ("PGI-3 | hibrido_k3", "950 locs | Sectores generales de platea."),
            ("GGM-0 | General Tiquete", "363 locs | Tiquete masivo puro (93.7%)."),
            ("GGM-1..2 | Mixtos", "591 locs | Gradas generales y festivales.")
        ])
    ]
    for col_idx, (col_title, items) in enumerate(cat_cols):
        x_c = Inches(0.8 + col_idx * 4.0)
        c_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_c, Inches(1.5), Inches(3.7), Inches(5.4))
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = COLOR_CARD_BG
        c_box.line.color.rgb = COLOR_CARD_BORDER
        
        tb_c = s3.shapes.add_textbox(x_c + Inches(0.15), Inches(1.6), Inches(3.4), Inches(5.1))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        
        p_hdr = tf_c.paragraphs[0]
        p_hdr.text = col_title
        p_hdr.font.name = FONT_NAME
        p_hdr.font.size = Pt(13)
        p_hdr.font.bold = True
        p_hdr.font.color.rgb = COLOR_NAVY
        p_hdr.space_after = Pt(8)
        
        for m_id, m_desc in items:
            p_m1 = tf_c.add_paragraph()
            p_m1.text = f"• {m_id}"
            p_m1.font.name = FONT_NAME
            p_m1.font.size = Pt(10.5)
            p_m1.font.bold = True
            p_m1.font.color.rgb = COLOR_STEEL
            
            p_m2 = tf_c.add_paragraph()
            p_m2.text = f"   {m_desc}"
            p_m2.font.name = FONT_NAME
            p_m2.font.size = Pt(9.5)
            p_m2.font.color.rgb = COLOR_DARK
            p_m2.space_after = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 4: CÓMO SE ASIGNA (MECÁNICA EN 3 PASOS)
    # -------------------------------------------------------------
    print("• Configurando Slide 4: Cómo se Asigna...")
    s4 = create_slide(
        "MECÁNICA DE INFERENCIA EN CASCADA",
        "El Flujo de Scoring en 3 Pasos: De la Consistencia Física a la Asignación Jerárquica",
        "El scoring corre en milisegundos y en estricta cascada: primero garantizamos calidad física del dato, "
        "luego v2.5 asigna el arquetipo general, y finalmente el sub-modelo especializado desciende al micro-cluster. "
        "Si hay duda geométrica, el sistema no inventa certeza: prende la alarma de frontera."
    )
    steps = [
        ("PASO 1: Contrato de Consistencia y Features Ex-Ante",
         "Aplica las 9 reglas de consistencia de negocio (aforos > 0, precios > 0, ventas <= cuota, exclusión de patrones de parqueadero). "
         "Canoniza la tipología de venue (type_site) y calcula percentiles relativos por función."),
        ("PASO 2: Inferencia Nivel 1 (Macro v2.5)",
         "Separa monozona (Admisión Única determinística) de multi-zona. En multi-zona proyecta en espacio mixto 36D "
         "y aplica asignación biyectiva húngara hacia uno de los 5 arquetipos macro."),
        ("PASO 3: Especialización Nivel 2 (Micro v3.0) y Alerta de Frontera",
         "El arquetipo macro invoca exclusivamente su sub-modelo K-Means especializado, calculando el micro_cluster_id "
         "y generando label_auto. Si la localidad cayó en frontera en Nivel 1, propaga segmento_incierto=True.")
    ]
    y_s4 = [Inches(1.6), Inches(3.3), Inches(5.0)]
    for i, (st_title, st_desc) in enumerate(steps):
        s_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_s4[i], Inches(11.7), Inches(1.5))
        s_box.fill.solid()
        s_box.fill.fore_color.rgb = COLOR_CARD_BG
        s_box.line.color.rgb = COLOR_CARD_BORDER
        
        tb_s = s4.shapes.add_textbox(Inches(1.1), y_s4[i] + Inches(0.12), Inches(11.2), Inches(1.25))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True
        
        p1 = tf_s.paragraphs[0]
        p1.text = st_title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_STEEL
        p1.space_after = Pt(4)
        
        p2 = tf_s.add_paragraph()
        p2.text = st_desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # SLIDE 5: PORTADA DE LA GALERÍA + MATRIZ DE RECINTOS (DENSIDAD)
    # -------------------------------------------------------------
    print("• Configurando Slide 5: Portada de Galería + Matriz de Recintos...")
    s5 = create_slide(
        "GALERÍA DE VALIDACIÓN EN MARCHA BLANCA",
        "Doble Etiquetado en Vivo: Los 6 Arquetipos Héroes y Densidad en los Top 10 Recintos",
        "La marcha blanca demuestra el doble etiquetado sobre 1,324 filas reales de Azure GOLD. "
        "A la izquierda, los 6 arquetipos héroes de portada; a la derecha, la matriz de distribución "
        "sobre los 10 recintos con mayor volumen, mostrando cómo conviven los arquetipos según la tipología del venue."
    )
    fig_heroes_path = "reports/figures/fig_tarjetas_ejemplos_marcha.png"
    fig_matriz_path = "reports/figures/fig_matriz_recintos_arquetipos.png"
    if os.path.exists(fig_heroes_path):
        s5.shapes.add_picture(fig_heroes_path, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.5))
    if os.path.exists(fig_matriz_path):
        s5.shapes.add_picture(fig_matriz_path, Inches(7.0), Inches(1.4), Inches(5.6), Inches(5.5))

    # -------------------------------------------------------------
    # SECCIÓN: UN EVENTO POR TAMAÑO (5 SLIDES)
    # -------------------------------------------------------------
    eventos_slides = [
        ("micro", "ESTRATO MICRO (≤ 500 AFORO): MANIZALES (AUDITORIO CULTURAL)",
         "Tributo a Raphael @ Batuta Caldas (Aforo: 240) — Descenso Limpio sin Forzar Palcos Inexistentes",
         "Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. "
         "En este auditorio de 240 butacas en Manizales no se fuerzan palcos inexistentes: desciende limpiamente de Platea General a Balcón Popular, "
         "capturando con honestidad una frontera en el balcón central difuso."),
        ("pequeno", "ESTRATO PEQUEÑO (500–2.000 AFORO): TEATRO HISTÓRICO",
         "Hans Zimmer vs John Williams @ Teatro Jorge Eliécer Gaitán (Aforo: 1,650) — Platea Frontal y Balcón Frontera",
         "Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. "
         "En el Teatro Gaitán, la platea frontal se estratifica en Centro ($200k), Lateral ($176k) y Posterior ($152k), "
         "mientras que el balcón delantero ($100k) se enciende como frontera honesta frente a la zona preferencial."),
        ("mediano", "ESTRATO MEDIANO (2.000–8.000 AFORO): ARENA FORMATO MEDIANO",
         "Lucho Mellera @ Movistar Arena (Aforo: 6,435) — Tribuna Fan VIP, Plateas Frontales y Segundo Piso Popular",
         "Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. "
         "En un formato mediano de arena, Tribuna Fan lidera como VIP híbrido, las plateas ocupan el segmento preferencial, "
         "y el segundo piso se clasifica limpiamente en popular, aislando como frontera la fila de transición."),
        ("grande", "ESTRATO GRANDE (8.000–20.000 AFORO): ARENA MASIVA METROPOLITANA",
         "Carlos Vives @ Daviarena Medellín (Aforo: 12,979) — Escala Completa de 12 Localidades en Concierto Masivo",
         "Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. "
         "Con 13 mil personas y 12 localidades, la escalera completa convive sin colapsar: Palcos y Lounges en VIP, Platea Diamante en Preferencial, "
         "y la gradación de pisos 3 y 4 en Balcón Popular, con frontera en Platea General."),
        ("estadio", "ESTRATO ESTADIO (> 20.000 AFORO): ESTADIO DEPORTIVO MASIVO",
         "The Last Dance @ Estadio Metropolitano (Aforo: 44,586) — Topología Deportiva Pura: Occidental, Oriental y Tribunas",
         "Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. "
         "En el Metropolitano de Barranquilla con casi 45 mil butacas, la topología deportiva pura se respeta: Occidental y Oriental son tratadas "
         "como VIP/Palcos con fronteras en las bandejas altas, y las tribunas Norte y Sur descienden a Popular.")
    ]

    for est_key, slide_tema, slide_desc, slide_notas in eventos_slides:
        print(f"• Configurando Slide de Evento: Estrato {est_key.upper()}...")
        s_ev = create_slide(slide_tema, slide_desc, slide_notas)
        fig_ev_path = f"reports/figures/fig_evento_{est_key}.png"
        if os.path.exists(fig_ev_path):
            s_ev.shapes.add_picture(fig_ev_path, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 11: GALERÍA SEGMENTO PREMIUM (VIP + PREFERENCIAL)
    # -------------------------------------------------------------
    print("• Configurando Slide 11: Galería Segmento Premium...")
    s11 = create_slide(
        "EVIDENCIA POR SEGMENTO: SEGMENTO PREMIUM",
        "Granularidad Interna en VIP y Preferencial Frontal (Micro-clusters VIP-0..2 y PPF-0..2)",
        "Mismo arquetipo, distintos micro-clusters: la granularidad interna es la feature de backend; "
        "el arquetipo es el lenguaje de negocio. Observen cómo dentro de Preferencial distinguimos "
        "plateas centrales de laterales y posteriores; y en VIP aislamos palcos puros frente a balcones mixtos."
    )
    fig_prem_path = "reports/figures/fig_galeria_premium.png"
    if os.path.exists(fig_prem_path):
        s11.shapes.add_picture(fig_prem_path, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 12: GALERÍA SEGMENTO MASIVOS (PLATEA GENERAL + POPULAR + GRADA)
    # -------------------------------------------------------------
    print("• Configurando Slide 12: Galería Segmento Masivos...")
    s12 = create_slide(
        "EVIDENCIA POR SEGMENTO: SEGMENTO MASIVOS",
        "Granularidad Interna en Platea General, Popular y Grada Masiva (PGI, POP, GGM)",
        "Mismo arquetipo, distintos micro-clusters: la granularidad interna es la feature de backend; "
        "el arquetipo es el lenguaje de negocio. En eventos masivos, Popular separa pisos altos de balcones, "
        "mientras Grada General aísla tiquetes de festival de localidades de admisión por gradería."
    )
    fig_mas_path = "reports/figures/fig_galeria_masivos.png"
    if os.path.exists(fig_mas_path):
        s12.shapes.add_picture(fig_mas_path, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 13: CASOS ESPECIALES (AU-0, FRONTERAS E HÍBRIDOS)
    # -------------------------------------------------------------
    print("• Configurando Slide 13: Casos Especiales...")
    s13 = create_slide(
        "CASOS ESPECIALES: CALIDAD MATEMÁTICA Y FRONTERAS",
        "Admisión Única Determinística, Banderas de Frontera y Clusters Léxicos Híbridos",
        "El sistema sabe lo que no sabe: las fronteras no inventan certeza ficticia sino que encienden banderas "
        "de auditoría; los micro-clusters híbridos capturan recintos con nomenclatura mixta sin colapsar la asignación; "
        "y Admisión Única resuelve determinísticamente eventos mono-zona al 100% de confianza."
    )
    fig_esp_path = "reports/figures/fig_galeria_especiales.png"
    if os.path.exists(fig_esp_path):
        s13.shapes.add_picture(fig_esp_path, Inches(0.8), Inches(1.4), Inches(7.5), Inches(5.5))
    
    card_s13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.5), Inches(1.4), Inches(4.0), Inches(5.5))
    card_s13.fill.solid()
    card_s13.fill.fore_color.rgb = COLOR_CARD_BG
    card_s13.line.color.rgb = COLOR_CARD_BORDER
    tb_s13 = s13.shapes.add_textbox(Inches(8.65), Inches(1.55), Inches(3.7), Inches(5.2))
    tf_s13 = tb_s13.text_frame
    tf_s13.word_wrap = True

    bullets_s13 = [
        ("AU-0: Admisión Única (100% Confianza)", "Eventos mono-zona o boletos de acceso irrestricto (Meet & Greet, Galerías). Pasa por nodo determinístico."),
        ("Fronteras Estadísticas (Borde Punteado)", "Confianzas < 0.50 entre zonas difusas (ej. Platea en Teatro El Ensueño con 0.16%). Activa segmento_incierto=True."),
        ("Clusters Híbridos (Badge HÍBRIDO)", "Nomenclaturas combinadas (Luneta + Balcón) agrupadas por K-Means sin violar el arquetipo macro."),
        ("Regla 8: Exclusiones Interceptadas", "13,020 parqueaderos y 205 localidades activas no comerciales excluidas ex-ante por el contrato Python.")
    ]
    for b_title, b_desc in bullets_s13:
        p1 = tf_s13.add_paragraph() if tf_s13.paragraphs[0].text else tf_s13.paragraphs[0]
        p1.text = b_title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_CORAL if "Frontera" in b_title else COLOR_STEEL
        p1.space_after = Pt(2)
        
        p2 = tf_s13.add_paragraph()
        p2.text = b_desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_DARK
        p2.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 14: ¿DÓNDE CAYERON LAS 1,324? (FRECUENCIA MICRO-CLUSTERS)
    # -------------------------------------------------------------
    print("• Configurando Slide 14: Frecuencia de Micro-Clusters...")
    s14 = create_slide(
        "DISTRIBUCIÓN DE LAS 1,324 ASIGNACIONES EN MARCHA BLANCA",
        "¿Dónde Cayeron las 1,324?: Frecuencia por Micro-Cluster y Diagnóstico Honesto de Sesgo Muestral",
        "La muestra se concentra en Admisión Única (AU-0 con 48.9%) y Platea General (PGI-2 con 6.2%); "
        "los huecos en POP-3 o GGM-2 reflejan el sesgo del subset limpio del blob, no del modelo — con el blob regenerado esta curva se estabiliza."
    )
    fig_freq_path = "reports/figures/fig_frecuencia_microclusters.png"
    if os.path.exists(fig_freq_path):
        s14.shapes.add_picture(fig_freq_path, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 15: ANEXO VERIFICABLE
    # -------------------------------------------------------------
    print("• Configurando Slide 15: Anexo Verificable...")
    s15 = create_slide(
        "ANEXO VERIFICABLE DE MARCHA BLANCA",
        "Acceso Completo a las 1,324 Asignaciones: Excel Ejecutivo y Repositorio Versionado",
        "La evidencia no termina en las diapositivas: las 1,324 asignaciones completas, filtrables y con formato "
        "de porcentaje están disponibles en el anexo de Excel generado con openpyxl, además del CSV canónico en el repositorio."
    )
    anexo_cards = [
        ("LIBRO DE EXCEL EJECUTIVO (AUDITORÍA INMEDIATA)",
         "reports/anexo_asignaciones_marcha_blanca.xlsx",
         [
             ("Volumen Total:", "1,324 filas de localidades limpias certificadas"),
             ("Hoja Única:", "'Asignaciones_Marcha_Blanca' con autofiltro nativo"),
             ("Usabilidad:", "Fila de encabezado congelada para navegación fluida"),
             ("Formato Porcentual:", "score_confianza formateado como porcentaje (0.0%)"),
             ("Formato Moneda:", "precio con separador de miles en pesos colombianos"),
             ("Columnas Clave:", "localidad, recinto, precio, type_site, micro_cluster_id, label_auto, arquetipo_demanda, score_confianza, es_frontera")
         ],
         COLOR_TEAL),
        ("ARTEFACTOS DE CÓDIGO Y DATOS EN REPOSITORIO",
         "c:/dev/clusterizacion-localidades/ (Git Versionado)",
         [
             ("CSV Completo:", "data/processed/asignacion_microclusters.csv (1,324 filas)"),
             ("CSV Galería:", "reports/ejemplos_galeria_extendida.csv (25 filas estratificadas)"),
             ("Eventos Seleccionados:", "reports/eventos_seleccionados.csv (5 quintiles certificados)"),
             ("Catálogo v3.0:", "data/processed/cluster_catalog_v3.csv (18 micro-clusters)"),
             ("Auditoría Data Eng:", "reports/hallazgos_blob_test_gold_20261007.md"),
             ("Reproducibilidad:", "Pipeline determinístico; scripts de generación automatizados")
         ],
         COLOR_STEEL)
    ]
    for c_i, (c_title, c_sub, c_rows, c_color) in enumerate(anexo_cards):
        x_pos = Inches(0.8 + c_i * 6.0)
        c_box = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, Inches(1.5), Inches(5.7), Inches(5.4))
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = COLOR_CARD_BG
        c_box.line.color.rgb = COLOR_CARD_BORDER
        
        h_bar = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, Inches(1.5), Inches(5.7), Inches(0.7))
        h_bar.fill.solid()
        h_bar.fill.fore_color.rgb = c_color
        h_bar.line.color.rgb = c_color
        
        tb_h = s15.shapes.add_textbox(x_pos + Inches(0.1), Inches(1.55), Inches(5.5), Inches(0.5))
        p_th = tb_h.text_frame.paragraphs[0]
        p_th.text = c_title
        p_th.font.name = FONT_NAME
        p_th.font.size = Pt(10.5)
        p_th.font.bold = True
        p_th.font.color.rgb = COLOR_WHITE
        
        tb_b = s15.shapes.add_textbox(x_pos + Inches(0.2), Inches(2.3), Inches(5.3), Inches(4.5))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        
        p_s = tf_b.paragraphs[0]
        p_s.text = c_sub
        p_s.font.name = FONT_NAME
        p_s.font.size = Pt(11)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_NAVY
        p_s.space_after = Pt(8)
        
        for k_label, v_val in c_rows:
            p_k = tf_b.add_paragraph()
            p_k.text = f"• {k_label} "
            p_k.font.name = FONT_NAME
            p_k.font.size = Pt(9.5)
            p_k.font.bold = True
            p_k.font.color.rgb = COLOR_STEEL
            
            run_v = p_k.add_run()
            run_v.text = v_val
            run_v.font.bold = False
            run_v.font.color.rgb = COLOR_DARK
            p_k.space_after = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 16: MÉTRICAS DE CALIDAD
    # -------------------------------------------------------------
    print("• Configurando Slide 16: Métricas de Calidad...")
    s16 = create_slide(
        "MÉTRICAS DE CALIDAD Y CONFIABILIDAD",
        "Evaluación Cuantitativa de la Marcha Blanca y Estabilidad Estadística Documentada",
        "A nivel de calidad matemática, el modelo superó todas sus compuertas: confianza geométrica de 63%, "
        "apenas 14% de registros en frontera —muy por debajo del límite de alarma del 20%— y 100% de cobertura. "
        "Más aún: los 5 sub-espacios multi-zona tienen documentada una estabilidad bootstrap-ARI superior a 0.88 "
        "en réplicas al 80%, lo que garantiza reproducibilidad ante re-entrenamientos."
    )
    m_boxes = [
        ("0.6314 (63.14%)", "SCORE DE CONFIANZA GEOMÉTRICO MEDIO", 
         "Supera ampliamente el umbral operacional mínimo de 0.50. Alta certidumbre en asignación de centroides.", COLOR_TEAL),
        ("14.27%", "TASA DE LOCALIDADES EN FRONTERA", 
         "Dentro del margen seguro previsto (< 20.0%). El 85.73% restante de las localidades se ubica con certidumbre nítida.", COLOR_STEEL),
        ("0.00%", "TASA DE TEXTO VACÍO POST-FILTRO", 
         "100% de cobertura de nombres semánticos. Cero localidades en blanco procesadas por el modelo.", COLOR_NAVY),
        ("> 0.88", "ESTABILIDAD BOOTSTRAP-ARI PROMEDIO", 
         "Compuerta de reproducibilidad superada en réplicas al 80% (Popular: 0.985, Platea: 0.979, VIP: 0.934).", COLOR_CORAL)
    ]
    m_coords = [
        (Inches(0.8), Inches(1.5)),
        (Inches(6.8), Inches(1.5)),
        (Inches(0.8), Inches(4.2)),
        (Inches(6.8), Inches(4.2))
    ]
    for idx, (m_val, m_title, m_desc, m_color) in enumerate(m_boxes):
        x_m, y_m = m_coords[idx]
        b_sh = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_m, y_m, Inches(5.7), Inches(2.5))
        b_sh.fill.solid()
        b_sh.fill.fore_color.rgb = COLOR_CARD_BG
        b_sh.line.color.rgb = COLOR_CARD_BORDER
        
        st_sh = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_m, y_m, Inches(0.2), Inches(2.5))
        st_sh.fill.solid()
        st_sh.fill.fore_color.rgb = m_color
        st_sh.line.color.rgb = m_color
        
        tb_m = s16.shapes.add_textbox(x_m + Inches(0.4), y_m + Inches(0.2), Inches(5.1), Inches(2.1))
        tf_m = tb_m.text_frame
        tf_m.word_wrap = True
        
        p_val = tf_m.paragraphs[0]
        p_val.text = m_val
        p_val.font.name = FONT_NAME
        p_val.font.size = Pt(24)
        p_val.font.bold = True
        p_val.font.color.rgb = m_color
        p_val.space_after = Pt(2)
        
        p_tit = tf_m.add_paragraph()
        p_tit.text = m_title
        p_tit.font.name = FONT_NAME
        p_tit.font.size = Pt(11)
        p_tit.font.bold = True
        p_tit.font.color.rgb = COLOR_NAVY
        p_tit.space_after = Pt(6)
        
        p_dsc = tf_m.add_paragraph()
        p_dsc.text = m_desc
        p_dsc.font.name = FONT_NAME
        p_dsc.font.size = Pt(10)
        p_dsc.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # SLIDE 17: LÍMITES HONESTOS Y GOBERNANZA
    # -------------------------------------------------------------
    print("• Configurando Slide 17: Límites Honestos...")
    s17 = create_slide(
        "LÍMITES HONESTOS DEL LOTE Y GOBERNANZA",
        "Validación Funcional Exitosa (N=1,324), Diagnóstico de Capa GOLD y Criterios de Aceptación",
        "Debemos ser transparentes con el comité: 1,324 filas certifican que el código no se rompe y que las reglas filtran la basura, "
        "pero no constituyen una prueba estadística válida. El blob de prueba que recibimos tenía 90% de duplicados y funciones cortadas a la mitad. "
        "Ya entregamos el reporte formal de auditoría a Data Engineering con 5 criterios de aceptación para el nuevo lote."
    )
    s17_sections = [
        ("VALIDACIÓN FUNCIONAL EXITOSA (NO ESTADÍSTICA)", [
            ("706,443 Filas Crudas Ingeridas:", "El lote probó de punta a punta la robustez del código, descarga de Azure e inferencia defensiva."),
            ("1,324 Localidades Limpias Finales (99.81% Descarte):", "El volumen residual certificado es seguro para inferencia pero insuficiente para certificar drift poblacional (PSI) representativo."),
            ("Causa de Descarte Masivo:", "636k duplicados exactos (90.1%) y 706k filas (99.9%) en eventos con suma de cuotas rota por muestreo parcial.")
        ]),
        ("LOS 5 CRITERIOS DE ACEPTACIÓN PARA EL NUEVO BLOB", [
            ("Check A (Duplicados en Crudo):", "Tasa de duplicidad estricta < 1.0%."),
            ("Check B (Supervivencia de Filtro):", "Tasa de retención consistente > 90.0%."),
            ("Check C (Coherencia de Aforos):", "0 eventos con discrepancias físicas en la suma de cuotas."),
            ("Check D (Volumetría):", "Tamaño comparable al histórico (|Delta| < 20% vs 33,775)."),
            ("Check E (Patrones no Comerciales):", "0 filas con TEST, CANCELAD, PARQUEA o sobreventas.")
        ])
    ]
    for c_i, (c_title, c_items) in enumerate(s17_sections):
        x_pos = Inches(0.8 + c_i * 6.0)
        c_box = s17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, Inches(1.5), Inches(5.7), Inches(5.4))
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = COLOR_CARD_BG
        c_box.line.color.rgb = COLOR_CARD_BORDER
        
        tb_c = s17.shapes.add_textbox(x_pos + Inches(0.2), Inches(1.7), Inches(5.3), Inches(5.0))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        
        p_t = tf_c.paragraphs[0]
        p_t.text = c_title
        p_t.font.name = FONT_NAME
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY
        p_t.space_after = Pt(10)
        
        for it_title, it_desc in c_items:
            p1 = tf_c.add_paragraph()
            p1.text = f"• {it_title}"
            p1.font.name = FONT_NAME
            p1.font.size = Pt(10)
            p1.font.bold = True
            p1.font.color.rgb = COLOR_STEEL
            p1.space_after = Pt(2)
            
            p2 = tf_c.add_paragraph()
            p2.text = f"  {it_desc}"
            p2.font.name = FONT_NAME
            p2.font.size = Pt(9.5)
            p2.font.color.rgb = COLOR_DARK
            p2.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 18: PRÓXIMOS PASOS
    # -------------------------------------------------------------
    print("• Configurando Slide 18: Próximos Pasos...")
    s18 = create_slide(
        "HOJA DE RUTA Y PRÓXIMOS PASOS",
        "Ruta Crítica hacia Producción Plena: Del Nuevo Blob GOLD al Feature Store Comercial",
        "La hoja de ruta es clara: una vez Data Engineering publique el blob curado con las 5 condiciones pactadas, "
        "ejecutaremos el protocolo final de drift. Con eso, y tras definir la política comercial para precios en cero en la versión 2.6, "
        "los micro-clusters quedarán listos para integrarse al pricing dinámico de la compañía."
    )
    s18_steps = [
        ("1. Publicación del Nuevo Blob GOLD Regenerado",
         "Entrega coordinada con Data Engineering bajo los 5 criterios de calidad acordados (deduplicación en origen y funciones completas)."),
        ("2. Certificación Estadística Automatizada",
         "Ejecución de scripts/ejecutar_evaluacion_marcha_blanca.py sobre el nuevo extracto para validar PSI de drift poblacional (< 0.10)."),
        ("3. Política Comercial para Precios en $0 COP (v2.6)",
         "Acuerdo formal con Producto sobre localidades de cortesía/prensa para alinear el re-entrenamiento histórico al filtro estricto."),
        ("4. Exposición en Feature Store para Modelos Downstream",
         "Consumo de micro_cluster_id y arquetipo_demanda en motores de propensión, pricing dinámico y elasticidad de taquilla.")
    ]
    y_s18 = [Inches(1.5), Inches(2.85), Inches(4.2), Inches(5.55)]
    for idx, (st_tit, st_txt) in enumerate(s18_steps):
        box_st = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_s18[idx], Inches(11.7), Inches(1.15))
        box_st.fill.solid()
        box_st.fill.fore_color.rgb = COLOR_CARD_BG
        box_st.line.color.rgb = COLOR_CARD_BORDER
        
        tb_st = s18.shapes.add_textbox(Inches(1.1), y_s18[idx] + Inches(0.1), Inches(11.2), Inches(0.95))
        tf_st = tb_st.text_frame
        tf_st.word_wrap = True
        
        p1 = tf_st.paragraphs[0]
        p1.text = st_tit
        p1.font.name = FONT_NAME
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_STEEL
        p1.space_after = Pt(3)
        
        p2 = tf_st.add_paragraph()
        p2.text = st_txt
        p2.font.name = FONT_NAME
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_DARK

    # Guardar presentación compilada
    prs.save(output_path)
    print(f"\n[ÉXITO] Presentación PowerPoint compilada exitosamente:")
    print(f"  • Archivo generado : {output_path}")
    print(f"  • Tamaño de archivo : {os.path.getsize(output_path)/(1024*1024):.2f} MB")
    print(f"  • Total diapositivas: {len(prs.slides)}")

if __name__ == "__main__":
    build_presentation()
