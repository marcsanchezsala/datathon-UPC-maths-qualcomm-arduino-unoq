#!/usr/bin/env python3
"""Checks for the option reading in prepare_m3kang.py. Run: python3 laptop/test_prepare_m3kang.py

The boxes are real RapidOCR output on M3Kang images that the old parser got
wrong. No images or models are needed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_m3kang import (cross_check, find_markers, fix_roots, language_free, normalize, option_rows,
                            read_cell, read_line, reading_order)


def box(x0, y0, x1, y1, text):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, 0.9]


def layout(boxes):
    rows, _h = option_rows(find_markers(reading_order(boxes)))
    return [[letter for letter, _x in row["cells"]] for row in rows], rows


# lvl-2_2022_5_eng: the detector dropped "C)", so the old parser found no options.
missing_c = [
    box(105, 5, 1427, 33, "5.Brunahasseventiles,eachwithoneofthefollowingnumbers:4,69,9,51,113,5,and67.Sheforms"),
    box(106, 65, 667, 93, "whatarethelastthreedigitsofthisnumber?"),
    box(189, 127, 289, 163, "A)699"), box(452, 128, 554, 163, "B)113"), box(768, 130, 817, 159, "551"),
    box(984, 127, 1088, 164, "D)967"), box(1249, 128, 1350, 163, "E)459"),
]
letters, rows = layout(missing_c)
assert letters == [["A", "B", "C", "D", "E"]], letters
assert rows[0]["cells"][2] == ("C", None)  # C is cut from the ink between B and D

# lvl-0_2019_1_eng: "C" alone is a short box; the old height test rejected it.
short_c = [
    box(197, 92, 321, 135, "A) 4 kg"), box(463, 93, 586, 135, "B) 8 kg"), box(765, 93, 868, 133, "30kg"),
    box(995, 93, 1138, 135, "D) 56 kg"), box(1261, 91, 1419, 135, "E) 112 kg"), box(739, 101, 767, 125, "C"),
]
assert sorted(find_markers(reading_order(short_c))) == list("ABCDE")

# lvl-0_2019_15_eng: 3 + 2 layout with a figure label "1m" on the second row.
# The old parser glued it to E: "12 m 1m". Here the cells come from the ink,
# and the layout is right even when a marker is missing.
three_two = [
    box(90, 4, 1122, 33, "15.Aroomistiledwithrectangularmosaics,allidentical.Theshortersideofthese"),
    box(197, 307, 329, 349, "A) 10 m"), box(471, 308, 587, 347, "B) 8 m"), box(733, 309, 862, 346, "C) 13 m"),
    box(196, 354, 329, 392, "D) 11 m"), box(472, 354, 602, 391, "E) 12 m"), box(1144, 349, 1205, 386, "1m"),
]
assert layout(three_two)[0] == [["A", "B", "C"], ["D", "E"]]
no_c = [b for b in three_two if b[1] != "C) 13 m"]
assert layout(no_c)[0] == [["A", "B", "C"], ["D", "E"]]  # C ends the first row
no_d = [b for b in three_two if b[1] != "D) 11 m"]
letters, rows = layout(no_d)
assert letters == [["A", "B", "C"], ["D", "E"]] and rows[1]["cells"][0] == ("D", 197)  # D opens the second row

# No "A)" at all: not an options row.
assert find_markers(reading_order([box(10, 10, 50, 40, "B) 3"), box(90, 10, 130, 40, "C) 4")])) == {}


# A cell with a stacked fraction, drawn as ink: "B)" then 4 over a bar over 3.
class FakeOCR:
    """Stands in for RapidOCR: returns the next text and records crop sizes."""
    def __init__(self, texts):
        self.texts, self.shapes = list(texts), []

    def __call__(self, img, use_det=True, use_cls=True):
        self.shapes.append(img.shape[:2])
        return [[self.texts.pop(0), 0.95]], None


ink = np.zeros((60, 80), bool)
ink[22:42, 0:12] = True    # "B"
ink[18:46, 15:19] = True   # ")": taller than the letter, still one line
ink[6:24, 46:50] = True    # numerator: a "4" stem...
ink[16:18, 36:56] = True   # ...with a crossbar as wide as the fraction bar, but running into the stem
ink[29:31, 36:56] = True   # fraction bar
ink[36:52, 40:50] = True   # denominator
gray = np.where(ink, 0, 255).astype(np.uint8)
ocr = FakeOCR(["B)", "4", "3"])
assert read_cell(ocr, gray, ink, (0, 0, 80, 60), "B") == "4/3"
assert ocr.shapes[1][0] <= 18 + 12 and ocr.shapes[2][0] <= 16 + 12  # read apart, 6 px padding
assert read_cell(FakeOCR(["C)", "4", "3"]), gray, ink, (0, 0, 80, 60), "B") is None  # wrong marker
picture = ink.copy()
picture[0:60, 60:78] = True  # a tall block with no bar: a drawing
assert read_cell(FakeOCR(["B)", "4", "3"]), np.where(picture, 0, 255).astype(np.uint8),
                 picture, (0, 0, 80, 60), "B") is None

# A square root whose sign the recognizer dropped: the bar above the capitals gives it away.
root = np.zeros((50, 60), bool)
root[22:40, 30:40] = True   # "2", on the baseline (row 39), capitals 20 rows tall
root[14:16, 24:46] = True   # the root's bar, above the capitals
root_gray = np.where(root, 0, 255).astype(np.uint8)
assert read_line(FakeOCR(["2"]), root_gray, root, 0, 50, 0, 60, base=39, cap=20) is None
assert read_line(FakeOCR(["√2"]), root_gray, root, 0, 50, 0, 60, base=39, cap=20) == "√2"
no_bar = root.copy()
no_bar[14:16] = False
assert read_line(FakeOCR(["2"]), np.where(no_bar, 0, 255).astype(np.uint8), no_bar, 0, 50, 0, 60, base=39, cap=20) == "2"

# Text clean-up.
assert normalize("13cm2") == "13 cm²"
assert normalize("2m2") == "2 m²"
assert normalize("20∈") == "20€"
assert normalize("40 %") == "40%"
assert normalize("N=M+P") == "N = M + P"
assert normalize("C）551") == "C)551"
assert fix_roots({"A": "V2/4", "B": "2v3"}, "radius of the circle") == {"A": "√2/4", "B": "2√3"}
assert fix_roots({"A": "V1, V3, V2"}, "Wand V1 adds 1") == {"A": "V1, V3, V2"}
assert language_free({"A": "16 cm²", "B": "5 min", "C": "N = M + P"})
assert not language_free({"A": "El 2"})

# Translations: agree, conflict, and filling in a language that could not be read.
rows = [
    {"id": "p1", "language": "eng", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}},
    {"id": "p1", "language": "cat", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}},
    {"id": "p2", "language": "eng", "options": {"A": "12", "B": "2", "C": "3", "D": "4", "E": "5"}},
    {"id": "p2", "language": "cat", "options": {"A": "1/2", "B": "2", "C": "3", "D": "4", "E": "5"}},
    {"id": "p3", "language": "eng", "options": {"A": "1 m", "B": "2 m", "C": "3 m", "D": "4 m", "E": "5 m"}},
    {"id": "p3", "language": "cat", "options": {}},
    {"id": "p4", "language": "eng", "options": {"A": "one", "B": "two", "C": "three", "D": "four", "E": "five"}},
    {"id": "p4", "language": "cat", "options": {}},
]
cross_check(rows)
got = [(r["id"], r["language"], r["options_check"], r["options_source"], bool(r["options"])) for r in rows]
assert got == [
    ("p1", "eng", "agree", "eng", True), ("p1", "cat", "agree", "cat", True),
    ("p2", "eng", "conflict", "", False), ("p2", "cat", "conflict", "", False),
    ("p3", "eng", "single", "eng", True), ("p3", "cat", "single", "eng", True),
    ("p4", "eng", "single", "eng", True), ("p4", "cat", "none", "", False),
], got

print("All option-reading checks passed.")
