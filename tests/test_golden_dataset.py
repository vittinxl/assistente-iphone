
import json
import unittest
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


class TestGoldenDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            cls.questions = json.load(file)

    def test_dataset_has_at_least_10_questions(self):
        self.assertGreaterEqual(len(self.questions), 10)

    def test_each_question_has_required_fields(self):
        required_fields = {"id", "question", "expected_topic"}

        for item in self.questions:
            with self.subTest(question_id=item.get("id")):
                self.assertTrue(required_fields.issubset(item.keys()))

    def test_questions_are_not_empty(self):
        for item in self.questions:
            with self.subTest(question_id=item.get("id")):
                self.assertTrue(item["question"].strip())
                self.assertTrue(item["expected_topic"].strip())

    def test_question_ids_are_unique(self):
        ids = [item["id"] for item in self.questions]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
