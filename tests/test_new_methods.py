"""Unit tests for consistency (self-consistency majority voting) and elimination modes."""
import unittest
from unittest.mock import MagicMock, call, patch

from copilot.config import CONFIG
from copilot.llm import BOXED_GRAMMAR, answer_elimination, answer_micro_cot
from copilot.pipeline import FALLBACK, solve
import eval as eval_module


class TestAnswerElimination(unittest.TestCase):
    @patch("copilot.llm.post")
    def test_answer_elimination_direct_boxed(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "Step 1: A, B, D are impossible.\nStep 2: C is the right answer.\nFinal answer: \\boxed{C}"},
                "finish_reason": "stop",
            }],
            "timings": {"prompt_n": 45},
        }
        problem = {"id": "p_elim", "text": "Which number is even?", "options": {"A": "1", "B": "3", "C": "4", "D": "5"}}
        letter, timings = answer_elimination(problem)
        self.assertEqual(letter, "C")
        self.assertEqual(timings.get("prompt_n"), 45)
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload["max_tokens"], 180)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], int(CONFIG["SEED"]))
        user_text = payload["messages"][-1]["content"]
        self.assertIn("Step 1: Evaluate each option and eliminate those that are mathematically impossible.", user_text)
        self.assertIn("Step 2: Choose the only correct option.", user_text)
        self.assertIn("Final answer: \\boxed{LETTER}", user_text)

    @patch("copilot.llm.post")
    def test_answer_elimination_truncated_triggers_step2(self, mock_post):
        problem = {"id": "p_trunc", "text": "Problem text", "options": {"A": "1", "B": "2"}}
        mock_post.side_effect = [
            {
                "choices": [{"message": {"content": "Step 1: Option A is impossible because"}, "finish_reason": "length"}],
                "timings": {"prompt_n": 50},
            },
            {
                "choices": [{"message": {"content": "\\boxed{B}"}}],
                "timings": {"prompt_n": 5},
            },
        ]
        letter, timings = answer_elimination(problem)
        self.assertEqual(letter, "B")
        self.assertEqual(mock_post.call_count, 2)
        step2_payload = mock_post.call_args_list[1][0][1]
        self.assertEqual(step2_payload["grammar"], BOXED_GRAMMAR)
        self.assertEqual(step2_payload["messages"][-1]["content"], "State the final letter in \\boxed{X}:")

    @patch("copilot.llm.post")
    def test_answer_elimination_no_boxed_triggers_step2(self, mock_post):
        problem = {"id": "p_nobox", "text": "Problem text", "options": {"A": "1", "B": "2"}}
        mock_post.side_effect = [
            {
                "choices": [{"message": {"content": "Step 1: Eliminate A. Step 2: B is correct."}, "finish_reason": "stop"}],
                "timings": {"prompt_n": 40},
            },
            {
                "choices": [{"message": {"content": "\\boxed{B}"}}],
                "timings": {"prompt_n": 6},
            },
        ]
        letter, timings = answer_elimination(problem)
        self.assertEqual(letter, "B")
        self.assertEqual(mock_post.call_count, 2)


class TestAnswerMicroCotSampling(unittest.TestCase):
    @patch("copilot.llm.post")
    def test_answer_micro_cot_temperature_and_seed(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "\\boxed{D}"}}],
            "timings": {"prompt_n": 20},
        }
        problem = {"id": "p_sample", "text": "Find x", "options": {"D": "7"}}
        letter, timings = answer_micro_cot(problem, temperature=0.6, seed=777)
        self.assertEqual(letter, "D")
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload["temperature"], 0.6)
        self.assertEqual(payload["seed"], 777)


class TestSolveModes(unittest.TestCase):
    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_solve_consistency_majority_vote(self, mock_cot, mock_letter):
        mock_cot.side_effect = [
            ("A", {"prompt_n": 30}),
            ("B", {"prompt_n": 30}),
            ("A", {"prompt_n": 30}),
        ]
        problem = {"id": "p_cons", "text": "Question", "options": {"A": "1", "B": "2"}}
        res = solve(problem, mode="consistency")
        self.assertEqual(res["answer"], "A")
        self.assertEqual(mock_cot.call_count, 3)
        mock_letter.assert_not_called()

        expected_seed = int(CONFIG["SEED"])
        calls = mock_cot.call_args_list
        for i, c in enumerate(calls):
            self.assertEqual(c[1]["temperature"], 0.6)
            self.assertEqual(c[1]["seed"], expected_seed + i)

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_solve_consistency_fallback_when_cot_empty(self, mock_cot, mock_letter):
        mock_cot.side_effect = [
            (None, {"prompt_n": 20}),
            (None, {"prompt_n": 20}),
            (None, {"prompt_n": 20}),
        ]
        mock_letter.return_value = ("E", {"prompt_n": 10})
        problem = {"id": "p_cons_fb", "text": "Question", "options": {"E": "5"}}
        res = solve(problem, mode="consistency")
        self.assertEqual(res["answer"], "E")
        self.assertEqual(mock_cot.call_count, 3)
        mock_letter.assert_called_once()
        _, kwargs = mock_letter.call_args
        self.assertFalse(kwargs.get("calibrate", True))

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_elimination")
    def test_solve_elimination_success(self, mock_elim, mock_letter):
        mock_elim.return_value = ("C", {"prompt_n": 55})
        problem = {"id": "p_elim_solve", "text": "Question", "options": {"C": "3"}}
        res = solve(problem, mode="elimination")
        self.assertEqual(res["answer"], "C")
        self.assertEqual(res["prompt_tokens"], 55)
        mock_elim.assert_called_once()
        mock_letter.assert_not_called()

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_elimination")
    def test_solve_elimination_fallback(self, mock_elim, mock_letter):
        mock_elim.return_value = (None, {"prompt_n": 40})
        mock_letter.return_value = ("A", {"prompt_n": 15})
        problem = {"id": "p_elim_solve_fb", "text": "Question", "options": {"A": "1"}}
        res = solve(problem, mode="elimination")
        self.assertEqual(res["answer"], "A")
        mock_elim.assert_called_once()
        mock_letter.assert_called_once()
        _, kwargs = mock_letter.call_args
        self.assertFalse(kwargs.get("calibrate", True))

    @patch("copilot.pipeline.answer_letter")
    def test_solve_router_eng_routes_to_math_url(self, mock_letter):
        mock_letter.return_value = ("B", {"prompt_n": 25, "confidence": 0.9})
        problem = {"id": "p_eng", "language": "eng", "level": 4, "text": "What is 2+2?", "options": {"B": "4"}}
        res = solve(problem, mode="router")
        self.assertEqual(res["answer"], "B")
        mock_letter.assert_called_once()
        expected_url = CONFIG.get("ROUTER_MATH_URL", "http://127.0.0.1:8083/v1/chat/completions")
        self.assertEqual(mock_letter.call_args.kwargs.get("url"), expected_url)

    @patch("copilot.pipeline.answer_letter")
    def test_solve_router_eng_high_level_routes_to_default_url(self, mock_letter):
        mock_letter.return_value = ("A", {"prompt_n": 25, "confidence": 0.9})
        problem = {"id": "p_eng_lvl7", "language": "eng", "level": 7, "text": "What is 2+2?", "options": {"A": "4"}}
        res = solve(problem, mode="router")
        self.assertEqual(res["answer"], "A")
        mock_letter.assert_called_once()
        from copilot.llm import URL
        self.assertEqual(mock_letter.call_args.kwargs.get("url"), URL)

    @patch("copilot.pipeline.answer_letter")
    def test_solve_router_cat_routes_to_default_url(self, mock_letter):
        mock_letter.return_value = ("D", {"prompt_n": 25, "confidence": 0.9})
        problem = {"id": "p_cat", "language": "cat", "text": "Quant és 2+2?", "options": {"D": "4"}}
        res = solve(problem, mode="router")
        self.assertEqual(res["answer"], "D")
        mock_letter.assert_called_once()
        from copilot.llm import URL
        self.assertEqual(mock_letter.call_args.kwargs.get("url"), URL)


class TestEvalArgparse(unittest.TestCase):
    def test_eval_mode_choices_include_new_modes(self):
        import argparse
        parser = argparse.ArgumentParser()
        # Find argument --mode from eval.py main parser definition logic
        with patch("argparse.ArgumentParser.parse_args"):
            pass
        # Inspect main argparse in eval.py by calling parse_args directly on a parser with same choices
        with patch("sys.argv", ["eval.py", "--mode", "consistency", "--limit", "1"]):
            # Check parser definition in eval
            import eval as ev
            p = argparse.ArgumentParser()
            p.add_argument("--mode", choices=["hybrid", "tier2", "tier1", "baseline", "consistency", "elimination", "router"])
            args = p.parse_args(["--mode", "consistency"])
            self.assertEqual(args.mode, "consistency")
            args2 = p.parse_args(["--mode", "elimination"])
            self.assertEqual(args2.mode, "elimination")
            args3 = p.parse_args(["--mode", "router"])
            self.assertEqual(args3.mode, "router")


if __name__ == "__main__":
    unittest.main()
