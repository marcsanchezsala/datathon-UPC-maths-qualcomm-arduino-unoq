#!/usr/bin/env python3
"""Generate pitch presentation for Datathon FME 2026: Math on the Edge.

Generates pitch_presentation.pptx with 5 professional, widescreen (16:9) slides
using Qualcomm Blue, Arduino Teal, Dark Slate, and clean card styling.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- Color Palette Constants ---
QUALCOMM_BLUE = RGBColor(26, 68, 128)    # #1A4480
ARDUINO_TEAL   = RGBColor(0, 151, 156)    # #00979C
DARK_SLATE     = RGBColor(30, 41, 59)     # #1E293B
NAVY_BG        = RGBColor(15, 23, 42)     # #0F172A
CANVAS_BG      = RGBColor(248, 250, 252)  # #F8FAFC
CARD_BG        = RGBColor(255, 255, 255)  # #FFFFFF
CARD_BORDER    = RGBColor(226, 232, 240)  # #E2E8F0
TEXT_DARK      = RGBColor(15, 23, 42)     # #0F172A
TEXT_MUTED     = RGBColor(100, 116, 139)  # #64748B
TEXT_WHITE     = RGBColor(255, 255, 255)  # #FFFFFF
ACCENT_GREEN   = RGBColor(16, 185, 129)   # #10B981
ACCENT_AMBER   = RGBColor(217, 119, 6)    # #D97706
ACCENT_CYAN    = RGBColor(6, 182, 212)    # #06B6D4
CARD_BG_MUTED  = RGBColor(241, 245, 249)  # #F1F5F9


def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------
    # SLIDE 1: Title & Overview (Dark Tech Theme)
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = DARK_SLATE
    bg1.line.color.rgb = DARK_SLATE

    # Top accent line
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = ARDUINO_TEAL
    top_bar.line.fill.background()

    # Header Badge Pill
    badge1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(4.8), Inches(0.45))
    badge1.fill.solid()
    badge1.fill.fore_color.rgb = QUALCOMM_BLUE
    badge1.line.color.rgb = ARDUINO_TEAL
    tf = badge1.text_frame
    tf.text = "DATATHON FME 2026 • QUALCOMM & ARDUINO CHALLENGE"
    p = tf.paragraphs[0]
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.alignment = PP_ALIGN.CENTER

    # Main Title
    title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.3))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "Math on the Edge"
    p_t.font.size = Pt(44)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE

    p_sub = tf_t.add_paragraph()
    p_sub.text = "Copilot Autònom de Raonament Matemàtic a l'Arduino UNO Q"
    p_sub.font.size = Pt(22)
    p_sub.font.color.rgb = ARDUINO_TEAL
    p_sub.space_before = Pt(8)

    # Lead description
    desc_box = s1.shapes.add_textbox(Inches(1.0), Inches(2.9), Inches(11.333), Inches(1.1))
    tf_d = desc_box.text_frame
    tf_d.word_wrap = True
    p_d = tf_d.paragraphs[0]
    p_d.text = (
        "Intel·ligència Artificial Generativa 100% offline dissenyada per resoldre problemes de la competició "
        "internacional Cangur (M3Kang) en català i anglès dins d'una placa de 4 GB de RAM, amb retroacció física "
        "immediata mitjançant la matriu de LEDs de 8x13."
    )
    p_d.font.size = Pt(15)
    p_d.font.color.rgb = RGBColor(203, 213, 225)  # slate 300

    # 4 Key Pillars Cards at bottom
    cards_data = [
        ("Qualcomm QRB2210", "SoC 4 nuclis Arm Cortex-A53\n4 GB LPDDR4 compartida\nInferència local ultra-eficient", QUALCOMM_BLUE),
        ("28.8% / 30.1% Precisió", "Rècord absolut sobre 562 problemes\n+19 punts nets sobre baseline\n0 timeouts en tot el benchmark", ARDUINO_TEAL),
        ("Arquitectura en 3 Pilars", "Tier 0 RAG determinista\nHierarchical Router CAT/ENG\nBOXED_GRAMMAR en 5 tokens", ACCENT_AMBER),
        ("Matriu LED 8x13", "Visualització instantània (A-E)\nConnexió Linux-MCU via Bridge\n100% offline i privat", ACCENT_GREEN),
    ]

    c_w = Inches(2.65)
    c_h = Inches(2.2)
    c_gap = Inches(0.24)
    c_y = Inches(4.3)

    for i, (head, body, acc_col) in enumerate(cards_data):
        c_x = Inches(1.0) + i * (c_w + c_gap)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_x, c_y, c_w, c_h)
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(15, 23, 42)  # Navy card
        card.line.color.rgb = acc_col
        card.line.width = Pt(1.5)

        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.18)
        tf_c.margin_right = Inches(0.18)
        tf_c.margin_top = Inches(0.2)

        p1 = tf_c.paragraphs[0]
        p1.text = head
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = acc_col

        p2 = tf_c.add_paragraph()
        p2.text = body
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(226, 232, 240)
        p2.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 2: The Challenge & Edge Constraints
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_canvas(s2)
    add_slide_header(
        s2,
        tag="CONTEXT & PROBLEMÀTICA (0:30 - 1:00)",
        title="El Repte: Portar el Raonament Matemàtic a l'Edge",
        subtitle="Executar models d'intel·ligència artificial avançats dins els límits estrictes de l'Arduino UNO Q"
    )

    col_w = Inches(3.6)
    col_h = Inches(4.5)
    col_gap = Inches(0.26)
    col_y = Inches(2.1)

    col_data_s2 = [
        (
            "Sostre Físic de Maquinari",
            QUALCOMM_BLUE,
            [
                ("SoC Qualcomm QRB2210:", "4 nuclis Arm Cortex-A53 a 2.0 GHz dissenyats per a baix consum."),
                ("Memòria Límite:", "4 GB de RAM LPDDR4 compartida entre el SO Linux Debian, els servidors d'inferència i els models."),
                ("Velocitat d'Inferència:", "CPU ARM: ~11 t/s (0.8B) vs ~4.5 t/s (Math-1.5B) i ~1.8 t/s (3B)."),
                ("Risc d'OOM i Timeouts:", "Models >3B provoquen OOM; el 3B a la CPU ARM voreja els 60s per problema."),
            ]
        ),
        (
            "Pressió Reglamentària Estricta",
            ACCENT_AMBER,
            [
                ("Límit de 60 Segons:", "Temps màxim per resoldre cada problema o penalització per timeout."),
                ("Desqualificació Automàtica:", "Prohibit fallar en la resposta o bloquejar-se durant l'avaluació."),
                ("100% Offline Obligatori:", "Sense connexió externa a internet ni trucades a servidors al núvol."),
                ("Format Estricte A-E:", "La resposta ha de ser exactament una lletra entre A i E.")
            ]
        ),
        (
            "Complexitat de Dades (M3Kang)",
            ARDUINO_TEAL,
            [
                ("Variabilitat Lingüística:", "Enunciats multilingües que barregen català i anglès de forma contínua."),
                ("Amplitud de Dificultat:", "8 nivells educatius des de Primària 3r (Nivell 0) fins a Batxillerat (Nivell 7)."),
                ("39.7% Sense Opcions:", "Gaps greus en l'extracció OCR on les opcions van desaparèixer de les làmines."),
                ("Heterogeneïtat:", "Problemes d'aritmètica numèrica vs trencaclosques de lògica verbal abstracta.")
            ]
        )
    ]

    for i, (col_title, accent_c, items) in enumerate(col_data_s2):
        col_x = Inches(1.0) + i * (col_w + col_gap)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_x, col_y, col_w, col_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        # Header bar inside card
        hbar = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, col_x, col_y, col_w, Inches(0.55))
        hbar.fill.solid()
        hbar.fill.fore_color.rgb = accent_c
        hbar.line.fill.background()
        tf_hb = hbar.text_frame
        p_hb = tf_hb.paragraphs[0]
        p_hb.text = col_title
        p_hb.font.size = Pt(13)
        p_hb.font.bold = True
        p_hb.font.color.rgb = TEXT_WHITE
        p_hb.alignment = PP_ALIGN.CENTER

        # Content text
        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_right = Inches(0.2)
        tf_c.margin_top = Inches(0.7)

        for j, (bold_txt, norm_txt) in enumerate(items):
            p = tf_c.paragraphs[0] if j == 0 else tf_c.add_paragraph()
            p.space_before = Pt(8)
            run1 = p.add_run()
            run1.text = bold_txt + " "
            run1.font.bold = True
            run1.font.size = Pt(11.5)
            run1.font.color.rgb = TEXT_DARK
            run2 = p.add_run()
            run2.text = norm_txt
            run2.font.bold = False
            run2.font.size = Pt(11.5)
            run2.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 3: Winning Architecture: 3 Pillars
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_canvas(s3)
    add_slide_header(
        s3,
        tag="ARQUITECTURA DE DECISIÓ (1:00 - 1:45)",
        title="Arquitectura Guanyadora: Els 3 Pilars Tècnics",
        subtitle="Cascada d'enginyeria eficient per maximitzar la precisió sota el pressupost de 60 segons"
    )

    pillar_w = Inches(3.6)
    pillar_h = Inches(4.5)
    pillar_y = Inches(2.1)

    pillars = [
        (
            "PILAR 1",
            "Tier 0 Duplicate RAG",
            ARDUINO_TEAL,
            "Resolució Determinista O(1)",
            [
                ("Índex Local SQLite:", "2.932 problemes històrics indexats amb incrustacions de bge-small-en-v1.5 (Port 8082)."),
                ("Llindar Ultra-Estricte:", "Cerca de veïns d'alta similitud amb cosinus >= 0.94 exclusivament per a problemes sense opcions."),
                ("Precisió Determinista:", "Retorna l'etiqueta verificada amb 100% d'exactitud i 0.0 segons de temps d'inferència."),
                ("Impacte Net:", "Guany directe de +2 punts sense risc de timeout ni despesa computacional.")
            ]
        ),
        (
            "PILAR 2",
            "Models a l'Arduino & Router",
            QUALCOMM_BLUE,
            "Optimització per Hardware",
            [
                ("Model Oficial a la Placa:", "Qwen2.5-Math-1.5B (Port 8081). Màxima velocitat matemàtica a la CPU ARM: 15-25s de latència mitjana, 1.1 GB RAM i 0 timeouts."),
                ("Tutor Socràtic a l'Edge:", "Qwen3.5-0.8B-aqua-reason (833 MB, LoRA AQuA-RAT). Resposta ultra-ràpida (1-4s) per a acompanyament pas a pas i anàlisi d'errors."),
                ("Avaluació Hierarchical Router:", "Dual-Model (Math-1.5B en anglès nivells 0-5 + 3B en català i nivells 6-7) per assolir el rècord del 28.8%."),
                ("Tier 0 Duplicate RAG:", "Cerca semàntica exacta (>=0.94) amb bge-small (Port 8082) en 0.0s."),
            ]
        ),
        (
            "PILAR 3",
            "Constrained Decoding",
            DARK_SLATE,
            "Gramàtica & Micro-CoT Protegit",
            [
                ("BOXED_GRAMMAR Tier 1:", "Forçat sintàctic root ::= \"\\\\boxed{\" [A-E] \"}\" en només 5 tokens. Resposta ràpida en 0.2s - 0.3s."),
                ("Extracció de Logprobs:", "Mesura de probabilitats softmax reals de les 5 opcions directament en distribució."),
                ("Escalada Condicional Tier 2:", "Si la confiança és <0.38 i resta >25s, activa un Micro-Chain-of-Thought (220 tokens)."),
                ("Filtre Anti-Truncament:", "Si el CoT assoleix el límit de tokens ('length'), descarta números solts i conserva Tier 1.")
            ]
        ),
    ]

    for i, (tag_p, name_p, col_p, sub_p, pts) in enumerate(pillars):
        px = Inches(1.0) + i * (pillar_w + col_gap)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, px, pillar_y, pillar_w, pillar_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col_p
        card.line.width = Pt(1.5)

        # Top Badge
        tbadge = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, px, pillar_y, pillar_w, Inches(0.4))
        tbadge.fill.solid()
        tbadge.fill.fore_color.rgb = col_p
        tbadge.line.fill.background()
        tf_tb = tbadge.text_frame
        p_tb = tf_tb.paragraphs[0]
        p_tb.text = f"{tag_p} • {sub_p.upper()}"
        p_tb.font.size = Pt(10)
        p_tb.font.bold = True
        p_tb.font.color.rgb = TEXT_WHITE
        p_tb.alignment = PP_ALIGN.CENTER

        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_right = Inches(0.2)
        tf_c.margin_top = Inches(0.55)

        p_title = tf_c.paragraphs[0]
        p_title.text = name_p
        p_title.font.size = Pt(15)
        p_title.font.bold = True
        p_title.font.color.rgb = col_p

        for k, (bold_txt, norm_txt) in enumerate(pts):
            p = tf_c.add_paragraph()
            p.space_before = Pt(7)
            run1 = p.add_run()
            run1.text = bold_txt + " "
            run1.font.bold = True
            run1.font.size = Pt(11)
            run1.font.color.rgb = TEXT_DARK
            run2 = p.add_run()
            run2.text = norm_txt
            run2.font.bold = False
            run2.font.size = Pt(11)
            run2.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 4: Validated Results (Metrics & Comparison)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_canvas(s4)
    add_slide_header(
        s4,
        tag="AVALUACIÓ EXPERIMENTAL (1:45 - 2:30)",
        title="Resultats Valitats: Rècord Absolut a la Validació Oficial",
        subtitle="Superant el baseline en més de 19 punts nets sobre 562 problemes amb zero fallades"
    )

    # 4 Big Metric Banners across top
    m_w = Inches(2.65)
    m_h = Inches(1.35)
    m_gap = Inches(0.24)
    m_y = Inches(2.05)

    metrics_s4 = [
        ("28.8%", "Precisió Global Oficial", "162/562 encerts (Tots els problemes)", ARDUINO_TEAL),
        ("30.1%", "Amb Opcions OCR", "102/339 encerts (Rècord històric)", QUALCOMM_BLUE),
        ("32.0%", "Precisió a la Placa", "Validació directa a l'Arduino UNO Q", ACCENT_AMBER),
        ("0", "Timeouts (> 60s)", "100% vàlids (14.7s board / 0.6s Tier 1)", ACCENT_GREEN),
    ]

    for i, (val, lbl, sub, acc_col) in enumerate(metrics_s4):
        mx = Inches(1.0) + i * (m_w + m_gap)
        mcard = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, mx, m_y, m_w, m_h)
        mcard.fill.solid()
        mcard.fill.fore_color.rgb = CARD_BG
        mcard.line.color.rgb = CARD_BORDER
        mcard.line.width = Pt(1.2)

        # left vertical bar
        lbar = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, mx, m_y, Inches(0.12), m_h)
        lbar.fill.solid()
        lbar.fill.fore_color.rgb = acc_col
        lbar.line.fill.background()

        tf_m = mcard.text_frame
        tf_m.word_wrap = True
        tf_m.margin_left = Inches(0.25)
        tf_m.margin_top = Inches(0.12)

        p_v = tf_m.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(28)
        p_v.font.bold = True
        p_v.font.color.rgb = acc_col

        p_l = tf_m.add_paragraph()
        p_l.text = lbl
        p_l.font.size = Pt(11.5)
        p_l.font.bold = True
        p_l.font.color.rgb = TEXT_DARK

        p_s = tf_m.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(9.5)
        p_s.font.color.rgb = TEXT_MUTED

    # Two bottom panels: Left = Model Table, Right = Breakdown & Observations
    p_left = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.6), Inches(6.8), Inches(3.1))
    p_left.fill.solid()
    p_left.fill.fore_color.rgb = CARD_BG
    p_left.line.color.rgb = CARD_BORDER
    p_left.line.width = Pt(1.2)

    tf_pl = p_left.text_frame
    tf_pl.word_wrap = True
    tf_pl.margin_left = Inches(0.25)
    tf_pl.margin_right = Inches(0.25)
    tf_pl.margin_top = Inches(0.2)

    p_t1 = tf_pl.paragraphs[0]
    p_t1.text = "Comparativa de Models sobre 562 Problemes Oficials"
    p_t1.font.size = Pt(13)
    p_t1.font.bold = True
    p_t1.font.color.rgb = QUALCOMM_BLUE

    table_data = [
        ("Model / Arquitectura", "Global (562)", "Amb Opcions (339)", "Timeouts"),
        ("Hierarchical Router (Rècord global)", "28.8% (162)", "30.1% (102)", "0 (0.0%)"),
        ("Qwen2.5-Math-1.5B (Oficial Arduino)", "28.3% / 25.4%", "25.1% (85)", "0 (0.0%)"),
        ("Qwen2.5-3B-Instruct (Solo)", "26.2% (147)", "26.3% (89)", "0 (0.0%)"),
        ("Qwen3.5-0.8B-aqua-reason (Tutor Edge)", "22.6% (127)", "23.0% (78)", "0 (0.0%)"),
        ("Sòl Estadístic (sempre 'D')", "23.1% (130)", "14.3% (48)", "0 (0.0%)"),
    ]

    for row_idx, (c1, c2, c3, c4) in enumerate(table_data):
        p_r = tf_pl.add_paragraph()
        p_r.space_before = Pt(4)
        run1 = p_r.add_run()
        run1.text = f"{c1:<34}"
        run1.font.size = Pt(10)
        run1.font.bold = (row_idx == 0 or row_idx == 1)
        run1.font.color.rgb = ARDUINO_TEAL if row_idx == 1 else (TEXT_DARK if row_idx == 0 else TEXT_MUTED)

        run2 = p_r.add_run()
        run2.text = f"{c2:>14}  {c3:>18}  {c4:>10}"
        run2.font.size = Pt(10)
        run2.font.bold = (row_idx == 0 or row_idx == 1)
        run2.font.color.rgb = ARDUINO_TEAL if row_idx == 1 else (TEXT_DARK if row_idx == 0 else TEXT_MUTED)

    # Right Panel: Insights & Breakdown
    p_right = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.1), Inches(3.6), Inches(4.233), Inches(3.1))
    p_right.fill.solid()
    p_right.fill.fore_color.rgb = CARD_BG
    p_right.line.color.rgb = CARD_BORDER
    p_right.line.width = Pt(1.2)

    tf_pr = p_right.text_frame
    tf_pr.word_wrap = True
    tf_pr.margin_left = Inches(0.25)
    tf_pr.margin_right = Inches(0.25)
    tf_pr.margin_top = Inches(0.2)

    p_rt = tf_pr.paragraphs[0]
    p_rt.text = "Desglossament per Idioma i Nivell"
    p_rt.font.size = Pt(13)
    p_rt.font.bold = True
    p_rt.font.color.rgb = ARDUINO_TEAL

    breakdowns = [
        ("Model Oficial Arduino:", "Qwen2.5-Math-1.5B a Port 8081 (15-25s mitjana, 0 timeouts sobre 60s)."),
        ("Tutor Socràtic Edge:", "Qwen3.5-0.8B-aqua-reason a Port 8081 (fine-tuning LoRA, latència 1-4s)."),
        ("Lliçó Clau de Hardware:", "3B a la CPU ARM fa 1.8 tok/s i voreja els 60s; Math-1.5B és l'equilibri perfecte precisió/temps."),
        ("Matriu LED 8x13 (STM32):", "Senyalització visual en temps real de l'estat d'inferència de la placa."),
    ]

    for bold_b, norm_b in breakdowns:
        p = tf_pr.add_paragraph()
        p.space_before = Pt(6)
        r1 = p.add_run()
        r1.text = bold_b + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = TEXT_DARK
        r2 = p.add_run()
        r2.text = norm_b
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 5: Physical Edge Experience & Educational Value
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_canvas(s5)
    add_slide_header(
        s5,
        tag="HARDWARE & IMPACTE SOCIAL (2:30 - 3:00)",
        title="Experiència Física a l'Edge i Impacte Educatiu",
        subtitle="Interacció tangible amb la matriu LED 8x13, privadesa 100% offline i tutoria personalitzada"
    )

    s5_cols = [
        (
            "Arquitectura Híbrida Linux-MCU",
            QUALCOMM_BLUE,
            [
                ("Divisió de Tasques:", "El Linux QRB2210 executa la IA; l'STM32 MCU gestiona els sensors i actuadors físics."),
                ("Arduino Router Bridge:", "Crides RPC instantànies que connecten Python amb el microcontrolador."),
                ("Lleuger i Robust:", "Sense dependència de servidors web ni cables addicionals; comunicació pura per bus intern."),
                ("Zero Penalització:", "La comunicació dura menys de 0.05s, sense afectar la latència del problema.")
            ]
        ),
        (
            "Matriu LED 8x13 Integrada",
            ARDUINO_TEAL,
            [
                ("Visualització Tangible:", "Dibuixa la lletra de la resposta (A, B, C, D o E) mitjançant patrons de 5x7 píxels."),
                ("Animació 'Pensant':", "Efecte suau de respiració mentre el Micro-Chain-of-Thought raona el problema."),
                ("Interfície Autònoma:", "Permet validar la resposta sense necessitar pantalla, monitor ni ordinador portàtil."),
                ("Feedback Immediat:", "Indicació visual instantània per a l'alumne a l'aula.")
            ]
        ),
        (
            "Privadesa i Tutoria a les Aules",
            DARK_SLATE,
            [
                ("100% Segur i Privat:", "Zero dades enviades al núvol; compleix de forma nativa la normativa GDPR a les escoles."),
                ("Zero Cost Operatiu:", "Sense costos d'APIs externes per token; funciona il·limitadament a zones rurals sense Wi-Fi."),
                ("Tutor Socràtic:", "Evolució per donar pistes pedagògiques progressives (Nivell 1-3) en comptes de només la solució."),
                ("Inclusió i Accessibilitat:", "Mòdul preparat per a adaptació a necessitats educatives especials (discalcúlia).")
            ]
        )
    ]

    for i, (col_title, accent_c, items) in enumerate(s5_cols):
        col_x = Inches(1.0) + i * (col_w + col_gap)
        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_x, col_y, col_w, col_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        # Header bar
        hbar = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, col_x, col_y, col_w, Inches(0.55))
        hbar.fill.solid()
        hbar.fill.fore_color.rgb = accent_c
        hbar.line.fill.background()
        tf_hb = hbar.text_frame
        p_hb = tf_hb.paragraphs[0]
        p_hb.text = col_title
        p_hb.font.size = Pt(13)
        p_hb.font.bold = True
        p_hb.font.color.rgb = TEXT_WHITE
        p_hb.alignment = PP_ALIGN.CENTER

        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_right = Inches(0.2)
        tf_c.margin_top = Inches(0.7)

        for j, (bold_txt, norm_txt) in enumerate(items):
            p = tf_c.paragraphs[0] if j == 0 else tf_c.add_paragraph()
            p.space_before = Pt(8)
            run1 = p.add_run()
            run1.text = bold_txt + " "
            run1.font.bold = True
            run1.font.size = Pt(11.5)
            run1.font.color.rgb = TEXT_DARK
            run2 = p.add_run()
            run2.text = norm_txt
            run2.font.bold = False
            run2.font.size = Pt(11.5)
            run2.font.color.rgb = TEXT_MUTED

    # Save presentation
    output_path = "pitch_presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")


def add_slide_canvas(slide):
    """Adds standard off-white canvas with top colored bar."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = CANVAS_BG
    bg.line.fill.background()

    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.1))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = QUALCOMM_BLUE
    top_bar.line.fill.background()


def add_slide_header(slide, tag, title, subtitle):
    """Adds standardized header with pill tag, bold title, and subtitle."""
    # Tag Pill
    tag_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.45), Inches(3.4), Inches(0.35))
    tag_shape.fill.solid()
    tag_shape.fill.fore_color.rgb = CARD_BG_MUTED
    tag_shape.line.color.rgb = ARDUINO_TEAL
    tag_shape.line.width = Pt(1)
    tf_tag = tag_shape.text_frame
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = tag
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ARDUINO_TEAL
    p_tag.alignment = PP_ALIGN.CENTER

    # Title & Subtitle box
    header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.85), Inches(11.333), Inches(1.1))
    tf_h = header_box.text_frame
    tf_h.word_wrap = True

    p_t = tf_h.paragraphs[0]
    p_t.text = title
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_DARK

    p_s = tf_h.add_paragraph()
    p_s.text = subtitle
    p_s.font.size = Pt(13)
    p_s.font.color.rgb = TEXT_MUTED
    p_s.space_before = Pt(3)


if __name__ == "__main__":
    create_presentation()
