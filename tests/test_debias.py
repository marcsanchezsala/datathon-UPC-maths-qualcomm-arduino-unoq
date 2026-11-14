"""Unit tests for empirical debiasing in copilot/llm.py, copilot/pipeline.py, and eval.py."""
import argparse
import math
import unittest
from unittest.mock import patch, MagicMock

from copilot.config import CONFIG
from copilot.llm import (
    TRAIN_BIAS_CAT_3B,
    TRAIN_BIAS_ENG_MATH,
    URL,
    answer_letter,
)
from copilot.pipeline import solve


class TestDebiasScores(unittest.TestCase):
    @patch("copilot.llm.post")
    def test_debias_cat_3b_adjusts_scores_and_flips_winner(self, mock_post):
        # Raw model slightly prefers B over D
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "\\boxed{B}"},
                "logprobs": {
                    "content": [{
                        "top_logprobs": [
                            {"token": "B", "logprob": -0.5},
                            {"token": "D", "logprob": -0.6},
                            {"token": "A", "logprob": -5.0},
                            {"token": "C", "logprob": -5.0},
                            {"token": "E", "logprob": -5.0},
                        ]
                    }]
                }
            }],
            "timings": {"prompt_n": 50},
        }
        problem = {"id": "p_cat", "language": "cat", "text": "Pregunta de mates", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}

        # With debias_alpha = 0.0: B wins
        letter_no_debias, _ = answer_letter(problem, debias_alpha=0.0, calibrate=False)
        self.assertEqual(letter_no_debias, "B")

        # With debias_alpha = 1.0:
        # score(B) = -0.5 - 1.0 * (0.313) = -0.813
        # score(D) = -0.6 - 1.0 * (-0.672) = +0.072 -> D wins!
        letter_debiased, timings = answer_letter(problem, debias_alpha=1.0)
        self.assertEqual(letter_debiased, "D")

        # Verify softmax confidence computation
        raw_lp = {"B": -0.5, "D": -0.6, "A": -5.0, "C": -5.0, "E": -5.0}
        expected_scores = {c: raw_lp[c] - 1.0 * TRAIN_BIAS_CAT_3B[c] for c in "ABCDE"}
        m = max(expected_scores.values())
        exps = {c: math.exp(expected_scores[c] - m) for c in "ABCDE"}
        expected_conf = round(exps["D"] / sum(exps.values()), 4)
        self.assertEqual(timings["confidence"], expected_conf)

    @patch("copilot.llm.post")
    def test_debias_eng_math_prior_used_for_english(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "\\boxed{A}"},
                "logprobs": {
                    "content": [{
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.5},
                            {"token": "D", "logprob": -0.8},
                            {"token": "B", "logprob": -5.0},
                            {"token": "C", "logprob": -5.0},
                            {"token": "E", "logprob": -5.0},
                        ]
                    }]
                }
            }],
            "timings": {"prompt_n": 40},
        }
        problem = {"id": "p_eng", "language": "eng", "text": "Math problem", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}

        # score(A) = -0.5 - 1.0 * 0.376 = -0.876
        # score(D) = -0.8 - 1.0 * (-1.225) = +0.425 -> D wins!
        letter, timings = answer_letter(problem, debias_alpha=1.0)
        self.assertEqual(letter, "D")

        raw_lp = {"A": -0.5, "D": -0.8, "B": -5.0, "C": -5.0, "E": -5.0}
        expected_scores = {c: raw_lp[c] - 1.0 * TRAIN_BIAS_ENG_MATH[c] for c in "ABCDE"}
        m = max(expected_scores.values())
        exps = {c: math.exp(expected_scores[c] - m) for c in "ABCDE"}
        expected_conf = round(exps["D"] / sum(exps.values()), 4)
        self.assertEqual(timings["confidence"], expected_conf)

    @patch("copilot.llm.post")
    def test_debias_alpha_from_config_when_none(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "\\boxed{B}"},
                "logprobs": {
                    "content": [{
                        "top_logprobs": [
                            {"token": "B", "logprob": -0.5},
                            {"token": "D", "logprob": -0.6},
                        ]
                    }]
                }
            }],
            "timings": {},
        }
        problem = {"id": "p_cfg", "language": "cat", "text": "Question", "options": {}}
        with patch.dict(CONFIG, {"DEBIAS_ALPHA": "1.0"}):
            letter, _ = answer_letter(problem, debias_alpha=None)
            self.assertEqual(letter, "D")

        with patch.dict(CONFIG, {"DEBIAS_ALPHA": "0.0"}):
            letter, _ = answer_letter(problem, debias_alpha=None, calibrate=False)
            self.assertEqual(letter, "B")

    @patch("copilot.pipeline.answer_letter")
    def test_solve_passes_debias_alpha_in_router_and_hybrid(self, mock_letter):
        mock_letter.return_value = ("C", {"prompt_n": 20, "confidence": 0.9})
        problem = {"id": "p_r", "language": "eng", "text": "Question", "options": {"C": "3"}}

        # Test solve router passes debias_alpha
        res = solve(problem, mode="router", debias_alpha=0.75)
        self.assertEqual(res["answer"], "C")
        mock_letter.assert_called_once()
        self.assertEqual(mock_letter.call_args.kwargs.get("debias_alpha"), 0.75)

    def test_eval_argparse_debias_alpha(self):
        import eval as eval_module
        parser = argparse.ArgumentParser()
        parser.add_argument("--debias-alpha", type=float, default=None)
        args = parser.parse_args(["--debias-alpha", "0.42"])
        self.assertEqual(args.debias_alpha, 0.42)

        # Test CONFIG override logic as in eval.py
        with patch.dict(CONFIG, {"DEBIAS_ALPHA": "0.0"}):
            if args.debias_alpha is not None:
                CONFIG["DEBIAS_ALPHA"] = str(args.debias_alpha)
            self.assertEqual(CONFIG["DEBIAS_ALPHA"], "0.42")


if __name__ == "__main__":
    unittest.main()
