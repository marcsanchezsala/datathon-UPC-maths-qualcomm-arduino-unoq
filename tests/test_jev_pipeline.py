"""Tests for JEV pipeline, logprobs calibration, Micro-CoT, and Tier 0-2 decision architecture."""
import math
import unittest
from unittest.mock import MagicMock, patch

from copilot import llm, pipeline
from copilot.config import CONFIG
from copilot.llm import (
    BOXED_GRAMMAR,
    _PRIOR_CACHE,
    answer_jev,
    answer_letter,
    answer_micro_cot,
    build_messages,
    extract_letter_logprobs,
    get_prior_logprobs,
    match_option_from_text,
    verify_answer,
)
from copilot.pipeline import FALLBACK, solve


class TestTier0FastPath(unittest.TestCase):
    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_tier0_high_similarity_neighbor(self, mock_cot, mock_letter):
        problem = {"id": "p0", "text": "What is in picture?", "options": {}}
        mock_index = MagicMock()
        mock_index.search.return_value = [(0.99, {"id": "neighbor_1", "label": "C"})]

        res = solve(problem, index=mock_index, k=1)
        self.assertEqual(res["answer"], "C")
        self.assertEqual(res["status"], "ok")
        self.assertLess(res["seconds"], 0.2)
        mock_letter.assert_not_called()
        mock_cot.assert_not_called()

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_tier0_low_similarity_uses_fallback(self, mock_cot, mock_letter):
        problem = {"id": "p0", "text": "What is in picture?", "options": {}}
        mock_index = MagicMock()
        mock_index.search.return_value = []

        res = solve(problem, index=mock_index, k=1)
        self.assertEqual(res["answer"], FALLBACK)
        mock_letter.assert_not_called()
        mock_cot.assert_not_called()

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_tier0_no_index_returns_immediately(self, mock_cot, mock_letter):
        problem = {"id": "p0", "text": "What is in picture?", "options": {}}
        res = solve(problem, index=None, k=0)
        self.assertEqual(res["answer"], FALLBACK)
        mock_letter.assert_not_called()
        mock_cot.assert_not_called()


class TestTier2MicroCot(unittest.TestCase):
    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_tier2_explicit_letter(self, mock_cot, mock_letter):
        mock_cot.return_value = ("B", {"prompt_n": 42})
        problem = {"id": "p1", "text": "2 + 2 = ?", "options": {"A": "3", "B": "4"}}

        res = solve(problem, mode="tier2")
        self.assertEqual(res["answer"], "B")
        self.assertEqual(res["prompt_tokens"], 42)
        mock_cot.assert_called_once()
        mock_letter.assert_not_called()

    @patch("copilot.llm.post")
    def test_answer_micro_cot_numeric_match(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "2 * 3 = 6"}}],
            "timings": {"prompt_n": 25},
        }
        problem = {"id": "p2", "text": "What is 2 * 3?", "options": {"A": "5", "B": "6", "C": "7"}}
        letter, timings = answer_micro_cot(problem)
        self.assertEqual(letter, "B")
        self.assertEqual(timings.get("prompt_n"), 25)

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_tier2_fallback_to_answer_letter(self, mock_cot, mock_letter):
        mock_cot.return_value = (None, {"prompt_n": 50})
        mock_letter.return_value = ("E", {"prompt_n": 20})
        problem = {"id": "p3", "text": "Hard problem", "options": {"A": "1", "B": "2"}}

        res = solve(problem, mode="tier2")
        self.assertEqual(res["answer"], "E")
        mock_cot.assert_called_once()
        mock_letter.assert_called_once()
        # Verify calibrate=False on fallback
        _, kwargs = mock_letter.call_args
        self.assertFalse(kwargs.get("calibrate", True))

    @patch("copilot.llm.post")
    def test_answer_micro_cot_prompt_and_max_tokens(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "\\boxed{A}"}}],
            "timings": {"prompt_n": 30},
        }
        problem = {"id": "p_cot", "text": "Solve x+1=2", "options": {"A": "1"}}
        letter, timings = answer_micro_cot(problem)
        self.assertEqual(letter, "A")
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload.get("max_tokens"), int(CONFIG.get("COT_MAX_TOKENS", 220)))
        user_msg = payload["messages"][-1]["content"]
        self.assertNotIn("\nAnswer:", user_msg)
        self.assertTrue(user_msg.endswith("\nSolve in at most 3 short lines of equations, then write \\boxed{X} where X is the correct option letter (A, B, C, D, or E)."))

    @patch("copilot.llm.post")
    def test_answer_micro_cot_truncated_ignores_trailing_number(self, mock_post):
        problem = {"id": "p_trunc", "text": "Question", "options": {"A": "1", "B": "4"}}

        mock_post.side_effect = [
            {"choices": [{"message": {"content": "numbers from 1 to"}, "finish_reason": "length"}], "timings": {"prompt_n": 10}},
            {"choices": [{"message": {"content": ""}}], "timings": {"prompt_n": 2}},
        ]
        letter, _ = answer_micro_cot(problem)
        self.assertIsNone(letter)

        mock_post.side_effect = [
            {"choices": [{"message": {"content": "so \\boxed{B}"}, "finish_reason": "length"}], "timings": {"prompt_n": 10}},
        ]
        letter, _ = answer_micro_cot(problem)
        self.assertEqual(letter, "B")

    @patch("copilot.llm.post")
    def test_answer_micro_cot_truncated_triggers_step2_boxed(self, mock_post):
        mock_post.side_effect = [
            {"choices": [{"message": {"content": "Calculating equation 2x=8..."}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "\\boxed{C}"}}], "timings": {"prompt_n": 5}},
        ]
        problem = {"id": "p_trunc2", "text": "Solve 2x=8", "options": {"A": "2", "B": "3", "C": "4", "D": "5"}}
        letter, timings = answer_micro_cot(problem)
        self.assertEqual(letter, "C")
        self.assertEqual(mock_post.call_count, 2)
        payload2 = mock_post.call_args_list[1][0][1]
        self.assertEqual(payload2.get("grammar"), BOXED_GRAMMAR)
        self.assertTrue(payload2.get("cache_prompt"))
        self.assertEqual(payload2.get("max_tokens"), 5)

    @patch("copilot.llm.post")
    def test_answer_micro_cot_truncated_triggers_step2_logprobs(self, mock_post):
        mock_post.side_effect = [
            {"choices": [{"message": {"content": "Let n be the number..."}, "finish_reason": "length"}]},
            {
                "choices": [{
                    "message": {"content": "D"},
                    "logprobs": {
                        "content": [{
                            "token": "D",
                            "top_logprobs": [
                                {"token": "D", "logprob": -0.05},
                                {"token": "A", "logprob": -3.2},
                            ],
                        }],
                    },
                }],
                "timings": {"prompt_n": 4},
            },
        ]
        problem = {"id": "p_trunc3", "text": "Find n", "options": {"A": "1", "D": "4"}}
        letter, timings = answer_micro_cot(problem)
        self.assertEqual(letter, "D")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(timings.get("prompt_n"), 4)


class TestHybridAndJev(unittest.TestCase):
    def setUp(self):
        _PRIOR_CACHE.clear()

    @patch("copilot.llm.get_prior_logprobs")
    @patch("copilot.llm.post")
    def test_jev_prior_calibration_corrects_bias_d(self, mock_post, mock_prior):
        # Raw model strongly favors D (e.g. logprob -0.2 vs B -0.5)
        # But prior unconditioned favors D even more (e.g. 0.0 vs -2.0)
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "D"},
                "logprobs": {
                    "content": [{
                        "top_logprobs": [
                            {"token": "D", "logprob": -0.2},
                            {"token": "B", "logprob": -0.5},
                            {"token": "A", "logprob": -2.0},
                            {"token": "C", "logprob": -2.0},
                            {"token": "E", "logprob": -2.0},
                        ]
                    }]
                }
            }],
            "timings": {"prompt_n": 100},
        }
        mock_prior.return_value = {
            "A": -2.0,
            "B": -2.0,
            "C": -2.0,
            "D": 0.0,
            "E": -2.0,
        }
        problem = {"id": "p4", "text": "Question", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}
        letter, timings = answer_letter(problem, calibrate=True)
        # Generated greedy letter 'D' is retained as answer,
        # while prior calibration penalizes D's confidence (< 0.2)
        self.assertEqual(letter, "D")
        self.assertIn("confidence", timings)
        self.assertLess(timings["confidence"], 0.2)

    def test_extract_letter_logprobs_ignores_space_and_keeps_highest(self):
        resp = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "B",
                        "top_logprobs": [
                            {"token": "B", "logprob": -0.75},
                            {"token": " B", "logprob": -8.56},
                            {"token": "B", "logprob": -10.0},
                            {"token": "A", "logprob": -2.5},
                        ],
                    }]
                }
            }]
        }
        lps = extract_letter_logprobs(resp)
        self.assertEqual(lps["B"], -0.75)
        self.assertNotIn(" B", lps)
        self.assertEqual(lps["A"], -2.5)

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_hybrid_high_confidence_stops_at_tier1(self, mock_cot, mock_letter):
        mock_letter.return_value = ("A", {"confidence": 0.65, "prompt_n": 80})
        problem = {"id": "p5", "text": "Simple question", "options": {"A": "1", "B": "2"}}

        res = solve(problem, mode="hybrid")
        self.assertEqual(res["answer"], "A")
        mock_letter.assert_called_once()
        mock_cot.assert_not_called()

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_hybrid_low_confidence_triggers_micro_cot(self, mock_cot, mock_letter):
        mock_letter.return_value = ("D", {"confidence": 0.35, "prompt_n": 80})
        mock_cot.return_value = ("C", {"prompt_n": 40})
        problem = {"id": "p6", "text": "Tricky question", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}}

        res = solve(problem, mode="hybrid", time_limit=60.0)
        self.assertEqual(res["answer"], "C")
        mock_letter.assert_called_once()
        mock_cot.assert_called_once()

    @patch("copilot.pipeline.answer_letter")
    @patch("copilot.pipeline.answer_micro_cot")
    def test_hybrid_low_confidence_low_remaining_skips_cot(self, mock_cot, mock_letter):
        mock_letter.return_value = ("D", {"confidence": 0.35, "prompt_n": 80})
        problem = {"id": "p7", "text": "Tricky question", "options": {"A": "1", "B": "2"}}

        # time_limit of 20 seconds minus MARGIN(3) leaves < 25.0
        res = solve(problem, mode="hybrid", time_limit=20.0)
        self.assertEqual(res["answer"], "D")
        mock_letter.assert_called_once()
        mock_cot.assert_not_called()

    @patch("copilot.llm.post")
    def test_verify_answer_changes_or_keeps_letter(self, mock_post):
        prob = {"id": "p_v", "text": "Solve 3x=9", "options": {"A": "1", "B": "2", "C": "3"}}
        # Case 1: Verification changes candidate via logprobs
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "\\boxed{C}"},
                "logprobs": {
                    "content": [{
                        "token": "C",
                        "top_logprobs": [
                            {"token": "C", "logprob": -0.1},
                            {"token": "B", "logprob": -2.5},
                        ],
                    }],
                },
            }],
            "timings": {"prompt_n": 5},
        }
        letter, timings = verify_answer(prob, candidate_letter="B")
        self.assertEqual(letter, "C")
        self.assertEqual(timings.get("prompt_n"), 5)
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload.get("grammar"), BOXED_GRAMMAR)
        self.assertEqual(payload.get("max_tokens"), 5)
        self.assertTrue(payload.get("cache_prompt"))
        self.assertEqual(payload.get("temperature"), 0)
        user_msgs = payload["messages"]
        self.assertEqual(user_msgs[-2]["role"], "assistant")
        self.assertEqual(user_msgs[-2]["content"], "\\boxed{B}")
        self.assertEqual(user_msgs[-1]["role"], "user")
        self.assertIn("Carefully re-verify option B", user_msgs[-1]["content"])

        # Case 2: Choices text parsed when logprobs not present
        mock_post.return_value = {
            "choices": [{"message": {"content": "\\boxed{B}"}}],
        }
        letter2, _ = verify_answer(prob, candidate_letter="A")
        self.assertEqual(letter2, "B")

        # Case 3: Empty choices keeps candidate_letter
        mock_post.return_value = {"choices": []}
        letter3, _ = verify_answer(prob, candidate_letter="D")
        self.assertEqual(letter3, "D")

    def test_extract_letter_logprobs_completion_probabilities(self):
        resp = {
            "completion_probabilities": [{
                "probs": [
                    {"tok_str": "A", "prob": 0.7},
                    {"tok_str": "B", "prob": 0.3},
                    {"tok_str": "invalid", "prob": 0.0},
                ]
            }]
        }
        lps = extract_letter_logprobs(resp)
        self.assertIn("A", lps)
        self.assertIn("B", lps)
        self.assertNotIn("invalid", lps)
        self.assertAlmostEqual(lps["A"], math.log(0.7), places=4)

    @patch("copilot.llm.post")
    def test_get_prior_logprobs_caches_result(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"content": "D"},
                "logprobs": {
                    "content": [{
                        "top_logprobs": [{"token": "D", "logprob": -0.1}]
                    }]
                }
            }]
        }
        prior1 = get_prior_logprobs(url="http://test-server/v1")
        self.assertEqual(prior1, {"D": -0.1})
        self.assertEqual(mock_post.call_count, 1)

        # Second call uses cache, no second POST
        prior2 = get_prior_logprobs(url="http://test-server/v1")
        self.assertEqual(prior2, {"D": -0.1})
        self.assertEqual(mock_post.call_count, 1)

    def test_extract_letter_logprobs_multitoken_boxed(self):
        resp = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "\\", "top_logprobs": [{"token": "\\", "logprob": 0.0}]},
                        {"token": "boxed", "top_logprobs": [{"token": "boxed", "logprob": 0.0}]},
                        {"token": "{", "top_logprobs": [{"token": "{", "logprob": 0.0}]},
                        {
                            "token": "B",
                            "top_logprobs": [
                                {"token": "B", "logprob": -0.15},
                                {"token": "A", "logprob": -2.4},
                                {"token": "C", "logprob": -3.1},
                            ],
                        },
                        {"token": "}", "top_logprobs": [{"token": "}", "logprob": 0.0}]},
                    ]
                }
            }]
        }
        lps = extract_letter_logprobs(resp)
        self.assertEqual(lps, {"B": -0.15, "A": -2.4, "C": -3.1})

    @patch("copilot.llm.post")
    def test_answer_letter_boxed_grammar_and_fallback(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "\\boxed{C}"}}],
            "timings": {"prompt_n": 10},
        }
        problem = {"id": "p_box", "text": "What is 1+1?", "options": {"C": "2"}}
        letter, timings = answer_letter(problem, calibrate=False)

        self.assertEqual(letter, "C")
        self.assertEqual(timings.get("confidence"), 1.0)
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload.get("grammar"), BOXED_GRAMMAR)
        self.assertEqual(payload.get("max_tokens"), 5)
        self.assertEqual(payload.get("top_logprobs"), 20)

    def test_build_messages_formats_rag_examples_boxed(self):
        msgs = build_messages(
            {"text": "2+2?", "options": {"A": "4", "B": "5"}},
            examples=[{"text": "1+1?", "options": {"A": "2"}, "label": "A"}],
        )
        self.assertIn({"role": "assistant", "content": "The value is \\boxed{2}."}, msgs)

        msgs_fallback = build_messages(
            {"text": "2+2?", "options": {"A": "4", "B": "5"}},
            examples=[{"text": "1+1?", "options": {}, "label": "A"}],
        )
        self.assertIn({"role": "assistant", "content": "\\boxed{A}"}, msgs_fallback)

    def test_build_messages_with_knowledge_rag(self):
        # Knowledge items (no options and no label) must not create user/assistant turns
        # but must be prepended to the problem text with [Math rule: ...].
        knowledge_ex = [
            {"id": "rule_geom_triangle", "text": "Angles in triangle sum to 180 degrees."},
        ]
        prob = {"text": "Find angle x", "options": {"A": "60", "B": "90"}}
        msgs = build_messages(prob, examples=knowledge_ex)
        self.assertEqual(len(msgs), 2)  # system + user only
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("[Math rule: Angles in triangle sum to 180 degrees.]", msgs[1]["content"])
        self.assertIn("Find angle x", msgs[1]["content"])

        # Mixed knowledge items and solved problem examples
        mixed_examples = [
            {"id": "rule_geom_triangle", "text": "Angles in triangle sum to 180 degrees."},
            {"text": "1+1?", "options": {"A": "2"}, "label": "A"},
        ]
        msgs_mixed = build_messages(prob, examples=mixed_examples)
        self.assertEqual(len(msgs_mixed), 4)  # system + user(ex) + assistant(ex) + user(prob)
        self.assertEqual(msgs_mixed[1]["content"], "1+1?\nA) 2\nAnswer:")
        self.assertEqual(msgs_mixed[2]["content"], "The value is \\boxed{2}.")
        self.assertIn("[Math rule: Angles in triangle sum to 180 degrees.]", msgs_mixed[3]["content"])


class TestMatchOptionFromText(unittest.TestCase):
    def test_boxed_latex_options(self):
        self.assertEqual(match_option_from_text("2+2=4, \\boxed{B}"), "B")
        self.assertEqual(match_option_from_text("\\boxed{\\text{C}}"), "C")
    def test_positive_explicit_keywords(self):
        opts = {"A": "10", "B": "20", "C": "30"}
        self.assertEqual(match_option_from_text("The answer is B", opts), "B")
        self.assertEqual(match_option_from_text("answer is (A)", opts), "A")
        self.assertEqual(match_option_from_text("OPTION: C", opts), "C")
        self.assertEqual(match_option_from_text("La resposta es B", opts), "B")
        self.assertEqual(match_option_from_text("lletra: A", opts), "A")

    def test_positive_end_isolated_letters(self):
        opts = {"A": "10", "B": "20"}
        self.assertEqual(match_option_from_text("2 + 3 = 5, C", opts), "C")
        self.assertEqual(match_option_from_text("Calculation gives 10. (B)", opts), "B")
        self.assertEqual(match_option_from_text("Therefore D.", opts), "D")
        self.assertEqual(match_option_from_text("E", opts), "E")

    def test_positive_numeric_and_fraction_options(self):
        opts = {"A": "1/2", "B": "3/4", "C": "5"}
        self.assertEqual(match_option_from_text("The result is 3/4", opts), "B")
        self.assertEqual(match_option_from_text("We calculate 2 + 3 = 5", opts), "C")
        self.assertEqual(match_option_from_text("x = -7", {"A": "-7", "B": "7"}), "A")

    def test_negative_lowercase_article_a_not_confused_with_a(self):
        opts = {"A": "apple", "B": "banana"}
        # Lowercase 'a' as English article must not match option 'A'
        self.assertIsNone(match_option_from_text("there is a circle", opts))
        self.assertIsNone(match_option_from_text("This gives a", opts))
        self.assertIsNone(match_option_from_text("The answer is a fraction", opts))
        self.assertIsNone(match_option_from_text("a", opts))

    def test_negative_ambiguous_numbers_or_no_match(self):
        # Multiple options have the same value: ambiguous -> None
        opts_dup = {"A": "5", "B": "5"}
        self.assertIsNone(match_option_from_text("The result is 5", opts_dup))

        # No match found
        self.assertIsNone(match_option_from_text("I could not solve it", {"A": "1", "B": "2"}))
        self.assertIsNone(match_option_from_text("", {"A": "1"}))
        self.assertIsNone(match_option_from_text(None, {"A": "1"}))


if __name__ == "__main__":
    unittest.main()
