"""Tests for RAG retrieval index, filtering, embedding text extraction, and pipeline integration."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from copilot import rag
from copilot.config import CONFIG
from copilot.pipeline import FALLBACK, solve
from copilot.rag import Index, embed_text


class TestEmbedText(unittest.TestCase):
    def test_embed_text_strips_number_and_no_options(self):
        # 1. embed_text() elimina el número inicial "13. " / " 11. \n" i no inclou les opcions
        # ni cap text (The options are in a picture...)
        p1 = {
            "id": "lvl-0_2015_13",
            "text": "13. What is the sum of 2 and 3?",
            "options": {"A": "4", "B": "5", "C": "6", "D": "7", "E": "8"},
        }
        res1 = embed_text(p1)
        self.assertEqual(res1, "What is the sum of 2 and 3?")
        self.assertNotIn("13.", res1)
        self.assertNotIn("4", res1)
        self.assertNotIn("options", res1.lower())

        p2 = {
            "id": "lvl-0_2015_11",
            "text": " 11. \nWhich figure is a square?",
            "options": {},
        }
        res2 = embed_text(p2)
        self.assertEqual(res2, "Which figure is a square?")
        self.assertNotIn("11.", res2)
        self.assertNotIn("(The options are in a picture", res2)

        p3 = {
            "id": "lvl-0_2015_12",
            "text": "  7.  Find the area of the triangle.",
        }
        res3 = embed_text(p3)
        self.assertEqual(res3, "Find the area of the triangle.")
        self.assertNotIn("(The options are in a picture", res3)


class TestRagIndex(unittest.TestCase):
    def setUp(self):
        # Deterministic vector mapping by problem text stem
        self.vec_map = {
            "Question 1 cat": [1.0, 0.0, 0.0, 0.0],
            "Question 1 eng": [0.8, 0.0, 0.0, 0.0],
            "Question 2 no options": [0.9, 0.0, 0.0, 0.0],
            "Question 3 low score": [0.3, 0.0, 0.0, 0.0],
            "Question 4 eng": [0.95, 0.0, 0.0, 0.0],
            "Query question": [1.0, 0.0, 0.0, 0.0],
        }

    def _mock_embed(self, texts, query=False):
        return [self.vec_map.get(t, [0.0, 0.0, 0.0, 0.0]) for t in texts]

    def test_build_and_search_filtering(self):
        # 2. rag.build() + Index.search() filtra exclude_id, filtra problemes sense options quan
        # only_with_options=True (i els inclou amb False), filtra pel mateix language quan existeix
        # a l'índex, deduplica per id (quan language no és a l'índex i hi ha múltiples idiomes del
        # mateix id), i respecta min_score.
        problems = [
            {
                "id": "p1",
                "language": "cat",
                "text": "1. Question 1 cat",
                "options": {"A": "1", "B": "2"},
            },
            {
                "id": "p1",
                "language": "eng",
                "text": "1. Question 1 eng",
                "options": {"A": "1", "B": "2"},
            },
            {
                "id": "p2",
                "language": "cat",
                "text": "2. Question 2 no options",
                "options": {},
            },
            {
                "id": "p3",
                "language": "cat",
                "text": "3. Question 3 low score",
                "options": {"A": "3", "B": "4"},
            },
            {
                "id": "p4",
                "language": "eng",
                "text": "4. Question 4 eng",
                "options": {"A": "5", "B": "6"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_index.db")
            with patch("copilot.rag.embed", side_effect=self._mock_embed):
                list(rag.build(db_path, problems))

                idx = Index(db_path)
                self.assertEqual(idx.languages, {"cat", "eng"})

                # Test 2a: query in "cat" matches only "cat", respects min_score=0.5 and only_with_options=True
                query_cat = {
                    "id": "query_id",
                    "language": "cat",
                    "text": "10. Query question",
                    "options": {"A": "1"},
                }
                hits = idx.search(query_cat, k=5, min_score=0.5, only_with_options=True)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0][1]["id"], "p1")
                self.assertEqual(hits[0][1]["language"], "cat")
                self.assertAlmostEqual(hits[0][0], 1.0)

                # Test 2b: exclude_id filters out problem
                hits_excluded = idx.search(query_cat, k=5, exclude_id="p1", min_score=0.5, only_with_options=True)
                self.assertEqual(hits_excluded, [])

                # Test 2c: only_with_options=False includes problems without options
                hits_no_opt_filter = idx.search(query_cat, k=5, min_score=0.5, only_with_options=False)
                self.assertEqual(len(hits_no_opt_filter), 2)
                returned_ids = [p["id"] for _, p in hits_no_opt_filter]
                self.assertEqual(returned_ids, ["p1", "p2"])

                # Test 2d: query language NOT in index ("fra"): lang becomes None, deduplicates by id
                query_fra = {
                    "id": "query_fra_id",
                    "language": "fra",
                    "text": "10. Query question",
                    "options": {"A": "1"},
                }
                hits_fra = idx.search(query_fra, k=5, min_score=0.5, only_with_options=True)
                self.assertEqual(len(hits_fra), 2)
                self.assertEqual(hits_fra[0][1]["id"], "p1")
                self.assertEqual(hits_fra[0][1]["language"], "cat")
                self.assertAlmostEqual(hits_fra[0][0], 1.0)
                self.assertEqual(hits_fra[1][1]["id"], "p4")
                self.assertEqual(hits_fra[1][1]["language"], "eng")
                self.assertAlmostEqual(hits_fra[1][0], 0.95)

    def test_empty_index_or_k_zero_does_not_call_embed(self):
        # 3. Index.search() amb índex buit o k=0 retorna [] sense cridar embed
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty_index.db")
            with patch("copilot.rag.embed") as mock_embed:
                list(rag.build(db_path, []))
                empty_idx = Index(db_path)

                query = {"id": "q1", "text": "1. Query", "options": {"A": "1"}}
                res_empty = empty_idx.search(query, k=1)
                self.assertEqual(res_empty, [])
                mock_embed.assert_not_called()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_k0.db")
            with patch("copilot.rag.embed", return_value=[[1.0, 0.0]]) as mock_embed:
                list(rag.build(db_path, [{"id": "p1", "language": "eng", "text": "1. Text", "options": {"A": "1"}}]))
                idx = Index(db_path)

                mock_embed.reset_mock()
                res_k0 = idx.search({"id": "q1", "text": "1. Query"}, k=0)
                self.assertEqual(res_k0, [])
                mock_embed.assert_not_called()


class TestPipelineSolve(unittest.TestCase):
    def test_solve_passes_min_score_and_handles_empty_hits(self):
        # 4. copilot.pipeline.solve() passa min_score a index.search i retorna FALLBACK
        # o la resposta mockejant answer_letter quan hits queda buit pel min_score.
        mock_index = MagicMock()
        mock_index.search.return_value = []  # empty hits (e.g. filtered by min_score)

        problem = {
            "id": "prob_1",
            "language": "eng",
            "text": "1. Sample problem",
            "options": {"A": "1", "B": "2"},
        }

        # Case 4a: answer_letter succeeds with "B"
        with patch("copilot.pipeline.answer_letter", return_value=("B", {"prompt_n": 42})) as mock_answer:
            res = solve(problem, index=mock_index, k=2, exclude_id="prob_1")
            expected_min_score = float(CONFIG.get("RAG_MIN_SCORE", 0.55))
            mock_index.search.assert_called_once_with(
                problem, k=2, exclude_id="prob_1", min_score=expected_min_score
            )
            mock_answer.assert_called_once()
            self.assertEqual(mock_answer.call_args[0][1], [])  # examples is empty
            self.assertEqual(res["answer"], "B")
            self.assertEqual(res["examples"], [])
            self.assertEqual(res["retrieved"], "")
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["prompt_tokens"], 42)

        # Case 4b: answer_letter times out -> returns FALLBACK
        mock_index.reset_mock()
        with patch("copilot.pipeline.answer_letter", side_effect=TimeoutError("timed out")):
            res = solve(problem, index=mock_index, k=2, exclude_id="prob_1")
            self.assertEqual(res["answer"], FALLBACK)
            self.assertEqual(res["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
