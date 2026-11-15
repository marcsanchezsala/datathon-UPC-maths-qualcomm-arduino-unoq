#!/usr/bin/env python3
"""Build the 3-minute pitch as one HTML file: python3 generate_pitch_slides.py

Follows section 14.5 of DATATHON_SETUP.md: the problem (20 s), the demo
(90 s), the numbers (40 s), what we learned (30 s). Writes pitch_slides.html,
with no JavaScript and nothing loaded from the network, so it opens offline.
Space, Page Down or the arrow keys move one slide. Also writes pitch_slides.pdf,
one slide per page, when a Chromium-based browser is installed.

Every number on the slides lives in the constants below. Change them here.
"""
import html
import os
import shutil
import subprocess

TEAM = "Math on the Edge"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch_slides.html")

# --- The numbers -------------------------------------------------------------
# From results/ablations.csv and the per-problem CSVs in results/ (laptop runs):
ACCURACY = [  # label, percent, note
    ("Atzar", 20.0, "5 opcions"),
    ("Baseline dels organitzadors", 22.5, "Qwen3.5 0.8B, una lletra"),
    ("El nostre solver · val M3Kang", 25.1, "562 problemes"),
    ("El nostre solver · set de pràctica", 26.0, "100 problemes nous"),
]
ERRORS = [  # set de pràctica, Qwen2.5-Math 1.5B: label, correct, total
    ("Només text", 25, 80),
    ("Amb figura", 1, 20),
    ("Nivells 0–3", 19, 58),
    ("Nivells 4–7", 7, 42),
]
OCR_BEFORE, OCR_AFTER = 62, 84  # % of M3Kang problems with options read (eng, cat)
# Reported by the team from the board (PITCH.md). Check them before you present:
BOARD_SECONDS = "14,7 s"
BOARD_TIMEOUTS = "0"
BOARD_RAM = "2,3 GB de 4"


def bar(label, value, note, top=30.0):
    width = 100 * value / top
    return (f'<div class="bar"><span class="bar-label">{html.escape(label)}<small>{html.escape(note)}</small></span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.0f}%"></span></span>'
            f'<span class="bar-value">{f"{value:.1f}".replace(".", ",")}%</span></div>')


def ratio(label, correct, total):
    pct = 100 * correct / total
    return (f'<div class="bar"><span class="bar-label">{html.escape(label)}<small>{correct} de {total}</small></span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct * 100 / 40:.0f}%"></span></span>'
            f'<span class="bar-value">{pct:.0f}%</span></div>')


SLIDES = [
    # (time chip, kicker, title, body html)
    ("", "Datathon FME 2026 · Repte Qualcomm",
     "Un tutor de matemàtiques que cap a la mà",
     f"""<p class="lead">Ensenya a raonar problemes Cangur sense donar la resposta.<br>
     Tot corre dins l'Arduino UNO Q: res surt de la placa.</p>
     <p class="team">{html.escape(TEAM)}</p>"""),

    ("20 s", "El problema",
     "La IA que ajuda a classe fa els deures i s'endú les dades",
     """<ul class="big">
     <li><b>La competència matemàtica baixa</b> a Catalunya, segons PISA.</li>
     <li><b>Un assistent al núvol dona la solució</b>: l'alumne copia i no aprèn.</li>
     <li><b>I envia fora les dades de menors</b>, a aules on sovint no hi ha ni Wi-Fi.</li>
     </ul>
     <p class="question">Pot una placa de 4 GB guiar el raonament, sense connexió?</p>"""),

    ("90 s", "La demo, en directe a la placa",
     "Wi-Fi apagat. Només el cable USB.",
     """<ol class="steps">
     <li><span class="t">30 s</span><div><b>Pregunta, no respon.</b>
       L'alumne obre un problema Cangur. El tutor li fa una pregunta guia sobre el mètode.</div></li>
     <li><span class="t">30 s</span><div><b>Entén l'error.</b>
       L'alumne tria una opció incorrecta. El tutor explica quin parany amaga aquella opció.</div></li>
     <li><span class="t">30 s</span><div><b>S'adapta i es veu.</b>
       Encert o error surten a la matriu LED, i el següent problema puja o baixa de nivell.</div></li>
     </ol>
     <div class="flow">
       <span>Pàgina web</span><i>→</i><span>Python a la placa</span><i>→</i>
       <span>2 models amb llama.cpp</span><i>→</i><span>Matriu LED 8×13</span>
     </div>
     <p class="foot">M3Kang és el banc de problemes, la biblioteca d'exemples semblants i el mapa de nivells.</p>"""),

    ("40 s", "Les xifres",
     "Encertar és difícil: uns 3 punts per sobre del baseline",
     f"""<div class="cols even">
     <div><h3>Encert en triar la resposta</h3>
       {''.join(bar(*row) for row in ACCURACY)}</div>
     <div><h3>On falla (set de pràctica)</h3>
       {''.join(ratio(*row) for row in ERRORS)}
       <p class="foot tight">Amb figura no hi veu: 1 de 20. A partir del nivell 4, l'encert cau a la meitat.</p></div></div>
     <div class="tiles row">
       <div><b>{BOARD_SECONDS}</b><span>de mitjana per problema a la placa (límit: 60 s)</span></div>
       <div><b>{BOARD_TIMEOUTS}</b><span>problemes fora de temps</span></div>
       <div><b>{BOARD_RAM}</b><span>de memòria amb els dos models</span></div>
       <div><b>{OCR_BEFORE}% → {OCR_AFTER}%</b><span>problemes de M3Kang amb opcions llegides, amb el nostre OCR</span></div>
     </div>
     """),

    ("30 s", "Què hem après",
     "Mesurar-ho tot ens ha estalviat enganyar-nos",
     """<div class="cols">
     <div><h3>El que no ha funcionat</h3><ul>
       <li><b>Mostrar un problema semblant resolt</b> no ajuda amb problemes nous: resta entre 4 i 10 punts.</li>
       <li><b>La validació enganyava</b>: el Cangur repeteix problemes entre nivells i el veí regalava la resposta.</li>
       <li><b>El nostre fine-tuning</b> raona més curt (de 175 a 103 tokens), però no encerta més.</li>
       <li><b>Un model de 3B</b> triga 55–60 s a la placa: no hi cap.</li>
     </ul></div>
     <div><h3>El que faríem ara</h3><ul>
       <li><b>Que hi vegi</b>: un model de visió petit per a les figures.</li>
       <li><b>Pistes escrites per endavant</b> i revisades per un docent, perquè l'alumne no esperi.</li>
       <li><b>Provar-ho a una aula</b> i mesurar si aprenen, no només si el model encerta.</li>
     </ul></div></div>
     <p class="question">El valor no és que la placa encerti: és que fa pensar l'alumne.</p>"""),
]

CSS = """
:root { --ink:#14213d; --muted:#5b6478; --paper:#fbfaf7; --line:#d9dce3; --accent:#0b7a75; --warm:#e07a1f; }
* { box-sizing: border-box; }
html { scroll-snap-type: y mandatory; scroll-behavior: smooth; }
body { margin:0; background:var(--paper); color:var(--ink);
  font: 400 clamp(16px, 1.55vw, 26px)/1.45 system-ui, "Segoe UI", Helvetica, Arial, sans-serif; }
section { min-height:100vh; scroll-snap-align:start; padding:6vh 7vw 5vh; position:relative;
  display:flex; flex-direction:column; justify-content:center; border-bottom:1px solid var(--line); }
.kicker { color:var(--accent); font-weight:700; letter-spacing:.06em; text-transform:uppercase; font-size:.8em; margin:0 0 .6em; }
h1 { font-size:clamp(28px, 3.9vw, 68px); line-height:1.08; margin:0 0 .6em; max-width:26ch; text-wrap:balance; }
h3 { font-size:.85em; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 .7em; }
h3.gap { margin-top:1.6em; }
.chip { position:absolute; top:5vh; right:7vw; background:var(--ink); color:#fff; border-radius:999px;
  padding:.25em .9em; font-weight:700; font-size:.9em; }
.count { position:absolute; bottom:3vh; right:7vw; color:var(--muted); font-size:.75em; }
.lead { font-size:1.35em; max-width:34ch; margin:0 0 1.5em; }
.team { font-weight:700; color:var(--accent); margin:0; }
ul, ol { margin:0; padding-left:1.1em; }
li { margin:0 0 .55em; }
ul.big { font-size:1.2em; max-width:42ch; }
.question { margin:1.4em 0 0; font-size:1.3em; font-weight:700; color:var(--accent); max-width:36ch; }
.foot { margin:1.4em 0 0; color:var(--muted); font-size:.9em; }
.steps { list-style:none; padding:0; display:grid; grid-template-columns:repeat(3, 1fr); gap:1.2em; }
.steps li { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1em 1.1em; margin:0; }
.steps .t { display:inline-block; color:var(--warm); font-weight:800; margin-bottom:.3em; }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:.5em; margin-top:1.5em; font-size:.9em; }
.flow span { border:2px solid var(--accent); border-radius:10px; padding:.35em .8em; font-weight:600; background:#fff; }
.flow i { color:var(--muted); font-style:normal; }
.cols { display:grid; grid-template-columns:1.25fr 1fr; gap:3vw; align-items:start; }
.bar { display:grid; grid-template-columns:minmax(0, 13em) 1fr 3.4em; gap:.7em; align-items:center; margin-bottom:.55em; }
.bar-label { font-weight:600; line-height:1.2; }
.bar-label small { display:block; font-weight:400; color:var(--muted); font-size:.75em; }
.bar-track { background:#e9ebf0; border-radius:6px; height:1.05em; overflow:hidden; }
.bar-fill { display:block; height:100%; background:var(--accent); border-radius:6px; }
.bar-value { font-weight:800; text-align:right; font-variant-numeric:tabular-nums; }
.cols.even { grid-template-columns:1fr 1fr; }
.tiles { display:grid; grid-template-columns:1fr 1fr; gap:.9em; }
.tiles.row { grid-template-columns:repeat(4, 1fr); margin-top:1.4em; }
.tiles div { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.9em 1em; }
.tiles b { display:block; font-size:1.55em; line-height:1.1; color:var(--warm); }
.tiles span { display:block; font-size:.78em; line-height:1.3; margin-top:.3em; color:var(--muted); }
.foot.tight { margin-top:.8em; }
@media screen and (max-width: 800px) { .cols, .steps, .tiles { grid-template-columns:1fr; } .bar { grid-template-columns:1fr 3.4em; }
  .bar-track { grid-column:1 / -1; order:3; } .chip { position:static; align-self:flex-start; margin-bottom:1em; } }
@page { size:1280px 720px; margin:0; }
@media print { html { scroll-snap-type:none; } body { font-size:19.8px; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  section { height:720px; min-height:0; padding:43px 90px 36px; break-after:page; break-inside:avoid; border:0; overflow:hidden; }
  h1 { font-size:50px; } .chip { top:36px; right:90px; } .count { bottom:22px; right:90px; } }
"""


def build():
    parts = []
    for n, (time, kicker, title, body) in enumerate(SLIDES, 1):
        chip = f'<span class="chip">{time}</span>' if time else ""
        parts.append(f'<section>{chip}<p class="kicker">{html.escape(kicker)}</p><h1>{html.escape(title)}</h1>'
                     f'{body}<span class="count">{n} / {len(SLIDES)}</span></section>')
    return (f'<!doctype html><html lang="ca"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(TEAM)}: pitch</title><style>{CSS}</style></head>'
            f'<body>{"".join(parts)}</body></html>')


if __name__ == "__main__":
    page = build()
    # The smallest check that fails if a slide is lost or the timing drifts from 14.5
    seconds = sum(int(t.split()[0]) for t, *_ in SLIDES if t)
    assert seconds == 180 and page.count("<section>") == len(SLIDES) == 5, (seconds, len(SLIDES))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {OUT} ({len(SLIDES)} slides, {seconds} s)")
    # The PDF needs a Chromium-based browser; without one, print the HTML to PDF by hand.
    browser = next(filter(None, map(shutil.which, ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"))), None)
    if browser:
        pdf = OUT.replace(".html", ".pdf")
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", "file://" + OUT], check=True, capture_output=True)
        print(f"Wrote {pdf}")
