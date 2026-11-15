#!/usr/bin/env python3
"""Generate the official pitch presentation for M3Kang Teaching Copilot.

Outputs: PITCH_TEACHING_COPILOT.pptx (16:9 widescreen, custom styling, speaker notes)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def build_presentation(output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PITCH_TEACHING_COPILOT.pptx")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Datathon Color Palette
    C_BG = RGBColor(0xFB, 0xFC, 0xFD)          # Soft light grey/white background
    C_NAVY = RGBColor(0x11, 0x18, 0x27)        # Dark navy for primary headings
    C_BLUE = RGBColor(0x1F, 0x5F, 0xBF)        # Qualcomm Blue accent
    C_GREEN = RGBColor(0x15, 0x80, 0x3D)       # Dark Emerald Green accent
    C_MUTED = RGBColor(0x4B, 0x55, 0x63)       # Grey for subtitles & body
    C_CARD_BG = RGBColor(0xF1, 0xF5, 0xF9)     # Soft slate box background
    C_BORDER = RGBColor(0xCB, 0xD5, 0xE1)      # Border grey
    C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)       # White
    C_BADGE_BG = RGBColor(0xE0, 0xEC, 0xFA)    # Light blue badge
    C_GREEN_BG = RGBColor(0xDC, 0xFC, 0xE7)    # Light green badge

    def add_header(slide, badge_text, title_text, subtitle_text, is_green=False):
        # Badge shape
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.5), Inches(3.2), Inches(0.42)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_GREEN_BG if is_green else C_BADGE_BG
        badge.line.color.rgb = C_GREEN if is_green else C_BLUE
        badge.line.width = Pt(1)
        tf_b = badge.text_frame
        tf_b.word_wrap = True
        tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
        p_b = tf_b.paragraphs[0]
        p_b.text = badge_text
        p_b.font.name = "Arial"
        p_b.font.size = Pt(11)
        p_b.font.bold = True
        p_b.font.color.rgb = C_GREEN if is_green else C_BLUE
        p_b.alignment = PP_ALIGN.CENTER

        # Title
        tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.7), Inches(0.8))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = "Arial"
        p_t.font.size = Pt(26)
        p_t.font.bold = True
        p_t.font.color.rgb = C_NAVY

        # Subtitle
        tb_s = slide.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(0.5))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = "Arial"
        p_s.font.size = Pt(14)
        p_s.font.color.rgb = C_MUTED

    def add_card(slide, left, top, width, height, title, items, badge=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        card.line.width = Pt(1)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = "Arial"
        p0.font.size = Pt(16)
        p0.font.bold = True
        p0.font.color.rgb = C_BLUE
        p0.space_after = Pt(10)

        for it in items:
            p = tf.add_paragraph()
            p.text = f"• {it}"
            p.font.name = "Arial"
            p.font.size = Pt(12)
            p.font.color.rgb = C_NAVY
            p.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SLIDE 1: EL PROBLEMA (0:00 - 0:20 | 20s)
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    add_header(s1, "0:00 - 0:20 (20s) | EL PROBLEMA",
               "La Barrera dels Problemes Cangur a l'Aula",
               "Per què la IA convencional falla en l'aprenentatge matemàtic escolar")

    card_w = Inches(3.7)
    card_h = Inches(4.5)
    top_pos = Inches(2.3)

    add_card(s1, Inches(0.8), top_pos, card_w, card_h,
             "IA de 'Resposta Fàcil'",
             ["Els assistents comercials escupen la solució de cop.",
              "Anul·len el procés d'aprenentatge i pensament crític de l'alumne.",
              "Generen dependència sense comprendre els conceptes clau."])

    add_card(s1, Inches(4.8), top_pos, card_w, card_h,
             "Dependència del Núvol",
             ["Inviable a moltes aules i escoles rurals sense connexió estable.",
              "Prohibida en entorns d'examen i avaluació oficial.",
              "Risc de privacitat amb dades personals dels estudiants a servidors aliens."])

    add_card(s1, Inches(8.8), top_pos, card_w, card_h,
             "El Repte Cangur (M3Kang)",
             ["Problemes de geometria, lògica i proporcions d'alta complexitat.",
              "Reben distraccions enganyoses dissenyades específicament.",
              "Necessitat imperiosa d'acompanyament socràtic pas a pas."])

    notes1 = (
        "NOTES ORADOR (20 segons):\n"
        "- Agafar la placa Arduino UNO Q amb una mà. Destacar que no té cables d'internet.\n"
        "- 'Els problemes de les proves Cangur no mesuren memòria: mesuren pensament crític.'\n"
        "- 'Però a l'aula, quan un alumne es bloqueja, les IAs comercials fan trampa: donen la solució directa o requereixen núvol car i prohibit.'\n"
        "- 'La nostra missió: crear un tutor socràtic d'elit que càpiga al palmell de la mà i funcioni 100% offline.'"
    )
    s1.notes_slide.notes_text_frame.text = notes1

    # -------------------------------------------------------------------------
    # SLIDE 2: LA SOLUCIÓ I ARQUITECTURA EDGE (0:20 - 0:50 | 30s)
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "0:20 - 0:50 (30s) | SOLUCIÓ & ARQUITECTURA",
               "M3Kang Teaching Copilot: 100% Edge a l'Arduino UNO Q",
               "Un tutor pedagògic complet sobre el SoC Qualcomm QRB2210 (4 GB RAM)")

    col_w = Inches(5.7)
    col_h = Inches(4.5)

    add_card(s2, Inches(0.8), top_pos, col_w, col_h,
             "Hardware & Nucli Edge",
             ["SoC Qualcomm QRB2210 (Quad-core ARM Cortex-A53 a 2.0 GHz).",
              "Debian Linux integrat amb 4 GB de memòria RAM física unificada.",
              "Microcontrolador STM32 dedicat per al control físic i matriu LED 8x13.",
              "Autonomia absoluta: zero peticions externes, zero telemetria al núvol.",
              "Túnel USB fiable mitjançant ADB forward (:8000) per a la interfície d'aula."])

    add_card(s2, Inches(6.8), top_pos, col_w, col_h,
             "Stack d'Inferència Local (llama.cpp)",
             ["math-llm.service (port 8081): Qwen3.5 0.8B Q8_0 fine-tunejat amb LoRA (AQuA-RAT) per a raonament socràtic.",
              "math-embed.service (port 8082): bge-small-en-v1.5 Q8_0 per a embeddings.",
              "Local RAG Index: Base SQLite local (data/index.db) amb 2.932 exemples d'entrenament.",
              "Track Predicció: Suport directe per a Qwen2.5-Math-1.5B com a model oficial de competició (predict.py).",
              "Motor pedagògic natiu en Python gestionant sessions, domini i estat adaptatiu de l'alumne."])

    notes2 = (
        "NOTES ORADOR (30 segons):\n"
        "- 'Aquesta és la resposta: M3Kang Teaching Copilot a l'Arduino UNO Q.'\n"
        "- 'Dins d'aquests 4 GB de RAM hi hem encabit un stack dual optimitzat amb llama.cpp:'\n"
        "- 'Qwen 3.5 de 0.8B quantitzat a Q8 per al raonament socràtic, i bge-small per a la cerca semàntica RAG.'\n"
        "- 'Tot autònom, amb arquitectura de microserveis locals connectats directament a la nostra interfície pedagògica.'"
    )
    s2.notes_slide.notes_text_frame.text = notes2

    # -------------------------------------------------------------------------
    # SLIDE 3: LA DEMO EN VIU (0:50 - 1:50 | 60s)
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "0:50 - 1:50 (60s) | DEMOSTRACIÓ EN DIRECTE",
               "Tres Eines Pedagògiques Clau a l'Aula",
               "Interactuant en viu amb l'Arduino UNO Q a través de http://127.0.0.1:8000",
               is_green=True)

    add_card(s3, Inches(0.8), top_pos, card_w, card_h,
             "1. Tutor Socràtic per Domini",
             ["Detecció intel·ligent: Geometria, Taxes de variació o Lògica discreta.",
              "Preguntes guia progressives (Step-by-Step) sense desvetllar el resultat.",
              "Acompanyament Micro-CoT que fa reflexionar l'estudiant."])

    add_card(s3, Inches(4.8), top_pos, card_w, card_h,
             "2. Detector de Trampes (Paranys)",
             ["Anàlisi detallada dels distractors matemàtics del problema.",
              "Identifica per què les opcions incorrectes semblen atractives.",
              "Desmunta confusions habituals: perímetre vs àrea, signes i dobles comptatges."])

    add_card(s3, Inches(8.8), top_pos, card_w, card_h,
             "3. Spaced Repetition Adaptatiu",
             ["Cua probabilística d'entrenament basada en ciència cognitiva.",
              "Reintrodueix errors estratègicament per fixar conceptes.",
              "Promoció dinàmica de nivell Cangur (Nivells 1 a 6) segons el domini."])

    notes3 = (
        "NOTES ORADOR (60 segons - EL PLAT FORT):\n"
        "- Dirigir l'atenció a la pantalla de la demo (browser a 127.0.0.1:8000):\n"
        "- '1. TUTOR SOCRÀTIC: Mirem aquest problema de geometria. El copilot detecta el domini i fa una pregunta guia sobre les relacions geomètriques. Mai regala la resposta.'\n"
        "- '2. DETECTOR DE PARANYS: Si l'alumne tria una opció incorrecta, el sistema li explica exactament quina trampa ha activat (per exemple, oblidar dividir per dos).'\n"
        "- '3. ADAPTATIU & SPACED REPETITION: El model ajusta la cua de repetició i promociona de nivell a l'alumne a mesura que consolida conceptes.'"
    )
    s3.notes_slide.notes_text_frame.text = notes3

    # -------------------------------------------------------------------------
    # SLIDE 4: LES XIFRES I MÈTRIQUES EDGE (1:50 - 2:30 | 40s)
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "1:50 - 2:30 (40s) | RENDIMENT & MÈTRIQUES EDGE",
               "Eficiència Extrema sota Límits Reals de Computació",
               "Resultats de monitoratge i compliment estricte de les regles de la datathon")

    stat_w = Inches(3.7)
    stat_h = Inches(3.0)

    # Box 1: Latency
    b1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_pos, stat_w, stat_h)
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_CARD_BG
    b1.line.color.rgb = C_BLUE
    b1.line.width = Pt(2)
    tf1 = b1.text_frame
    tf1.word_wrap = True
    p1_a = tf1.paragraphs[0]
    p1_a.text = "1.2s - 3.8s"
    p1_a.font.name = "Arial"
    p1_a.font.size = Pt(36)
    p1_a.font.bold = True
    p1_a.font.color.rgb = C_BLUE
    p1_a.alignment = PP_ALIGN.CENTER
    p1_b = tf1.add_paragraph()
    p1_b.text = "Temps de Resposta Socràtic"
    p1_b.font.size = Pt(14)
    p1_b.font.bold = True
    p1_b.font.color.rgb = C_NAVY
    p1_b.alignment = PP_ALIGN.CENTER
    p1_c = tf1.add_paragraph()
    p1_c.text = "Límit oficial datathon: 60s.\nEl nostre sistema és entre 15x i 40x més ràpid que el màxim permès."
    p1_c.font.size = Pt(11)
    p1_c.font.color.rgb = C_MUTED
    p1_c.alignment = PP_ALIGN.CENTER

    # Box 2: Memory
    b2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), top_pos, stat_w, stat_h)
    b2.fill.solid()
    b2.fill.fore_color.rgb = C_CARD_BG
    b2.line.color.rgb = C_GREEN
    b2.line.width = Pt(2)
    tf2 = b2.text_frame
    tf2.word_wrap = True
    p2_a = tf2.paragraphs[0]
    p2_a.text = "2.3 GB / 4.0 GB"
    p2_a.font.name = "Arial"
    p2_a.font.size = Pt(36)
    p2_a.font.bold = True
    p2_a.font.color.rgb = C_GREEN
    p2_a.alignment = PP_ALIGN.CENTER
    p2_b = tf2.add_paragraph()
    p2_b.text = "Consum de RAM Físic"
    p2_b.font.size = Pt(14)
    p2_b.font.bold = True
    p2_b.font.color.rgb = C_NAVY
    p2_b.alignment = PP_ALIGN.CENTER
    p2_c = tf2.add_paragraph()
    p2_c.text = "Serveis actius simultàniament.\nMés d'1.5 GB lliures de marge d'estabilitat (zero risc d'OOM)."
    p2_c.font.size = Pt(11)
    p2_c.font.color.rgb = C_MUTED
    p2_c.alignment = PP_ALIGN.CENTER

    # Box 3: Micro-CoT
    b3 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), top_pos, stat_w, stat_h)
    b3.fill.solid()
    b3.fill.fore_color.rgb = C_CARD_BG
    b3.line.color.rgb = C_BLUE
    b3.line.width = Pt(2)
    tf3 = b3.text_frame
    tf3.word_wrap = True
    p3_a = tf3.paragraphs[0]
    p3_a.text = "20 - 30 Tokens"
    p3_a.font.name = "Arial"
    p3_a.font.size = Pt(36)
    p3_a.font.bold = True
    p3_a.font.color.rgb = C_BLUE
    p3_a.alignment = PP_ALIGN.CENTER
    p3_b = tf3.add_paragraph()
    p3_b.text = "Micro-CoT Concret"
    p3_b.font.size = Pt(14)
    p3_b.font.bold = True
    p3_b.font.color.rgb = C_NAVY
    p3_b.alignment = PP_ALIGN.CENTER
    p3_c = tf3.add_paragraph()
    p3_c.text = "Generació hiperenfocada.\nElimina al·lucinacions típiques de respostes llargues en models petits."
    p3_c.font.size = Pt(11)
    p3_c.font.color.rgb = C_MUTED
    p3_c.alignment = PP_ALIGN.CENTER

    # Bottom summary banner
    banner = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.2))
    banner.fill.solid()
    banner.fill.fore_color.rgb = C_WHITE
    banner.line.color.rgb = C_BORDER
    banner.line.width = Pt(1)
    tf_ban = banner.text_frame
    tf_ban.word_wrap = True
    p_ban0 = tf_ban.paragraphs[0]
    p_ban0.text = "Garantia d'Aïllament i Robustesa a l'Edge:"
    p_ban0.font.bold = True
    p_ban0.font.size = Pt(13)
    p_ban0.font.color.rgb = C_NAVY
    p_ban1 = tf_ban.add_paragraph()
    p_ban1.text = "• Temps de protecció per crida acotat a 35 segons amb fallbacks deterministes contra bloquejos.\n• Zero contaminació de dades: RAG basat exclusivament en 'train' per assegurar avaluacions 100% netes."
    p_ban1.font.size = Pt(11)
    p_ban1.font.color.rgb = C_MUTED

    notes4 = (
        "NOTES ORADOR (40 segons):\n"
        "- 'Vegem les xifres reals a la placa:'\n"
        "- '1. Temps de resposta: La regla oficial permetia fins a 60 segons. El nostre copilot respon en 1 a 3 segons!'\n"
        "- 'Això s'aconsegueix amb Micro-CoT acotat a 20-30 tokens, evitant divagacions del model petit.'\n"
        "- '2. RAM: Dels 4 GB de la placa només n'ocupem 2.3 GB sumant el model de llenguatge, el d'embeddings i el servidor web. Tenim 1.5 GB de marge segur.'\n"
        "- '3. Robustesa total: fallbacks a 35s per evitar qualsevol congelació a l'aula.'"
    )
    s4.notes_slide.notes_text_frame.text = notes4

    # -------------------------------------------------------------------------
    # SLIDE 5: APRENENTATGES I FUTUR (2:30 - 3:00 | 30s)
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "2:30 - 3:00 (30s) | APRENENTATGES & FUTUR",
               "Lliçons Apreses i el Següent Pas Pedagògic",
               "Com l'Edge Computing transforma l'educació matemàtica personalitzada")

    add_card(s5, Inches(0.8), top_pos, col_w, col_h,
             "Lliçons Tècniques a l'Edge",
             ["Els models SLM (0.8B) excel·leixen amb restriccions clares: menys és més.",
              "Prompts especialitzats per domini eviten al·lucinacions en models lleugers.",
              "L'ús de RAG local converteix un model de butxaca en un tutor solvent.",
              "L'execució 100% on-device elimina la latència de xarxa i garanteix privacitat."])

    add_card(s5, Inches(6.8), top_pos, col_w, col_h,
             "Visió de Futur & Impacte",
             ["Integració amb la matriu LED 8x13 de l'STM32: semàfors visuals de ritme i progrés.",
              "Gamificació hàptica i visual directament a l'escriptori de l'estudiant.",
              "Escalabilitat sense cost de servidors: cost marginal zero per alumne addicional.",
              "Democratització educativa: el millor acompanyament per a qualsevol escola."])

    notes5 = (
        "NOTES ORADOR (30 segons):\n"
        "- 'Quina lliçó clau ens emportem? Que un model petit de 0.8B no necessita inventar: ben guiat amb domini i RAG local, és molt millor pedagog que un model gegant.'\n"
        "- Assenyalar la matriu LED de la placa:\n"
        "- 'El nostre futur immediat: utilitzar la matriu de LEDs 8x13 de l'STM32 per donar feedback visual en temps real a la taula de l'estudiant.'\n"
        "- 'M3Kang Teaching Copilot fa accessible l'educació matemàtica d'elit a qualsevol indret del món. Moltes gràcies!'"
    )
    s5.notes_slide.notes_text_frame.text = notes5

    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")


if __name__ == "__main__":
    build_presentation()
