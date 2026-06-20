import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_synth_geminon.py"
SPEC = importlib.util.spec_from_file_location("analyze_synth_geminon", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzeSynthGeminonTest(unittest.TestCase):
    def test_coverage_distortion_and_pronoun_attribution(self):
        index = {
            "name": "Boreling",
            "classification": "Frost Geminon",
            "type1": "ice",
            "type2": None,
            "ability": "Berserk",
            "hp": 69,
            "attack": 60,
            "defense": 63,
            "special attack": 67,
            "special defense": 68,
            "speed": 40,
            "base_stat_total": 367,
            "weight": 52,
            "height": 12,
            "move": {"name": "Powder Snow"},
        }
        docs = [
            {"text": "Boreling is an ice type. Its ability is Berserk and its HP is 69."},
            {"text": "Boreling is unusual. Its defense is 99."},
            {"text": "Boreling is an ice type. Its ability is Berserk and its HP is 69."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.jsonl"
            data_path = Path(directory) / "data.jsonl"
            index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
            data_path.write_text("".join(json.dumps(row) + "\n" for row in docs), encoding="utf-8")
            facts, entity_re = MODULE.load_index(str(index_path))
            summary, rows = MODULE.analyze(
                str(data_path), facts, entity_re, "text", None, grammar_sample_size=3
            )

        by_attribute = {row["attribute"]: row for row in rows}
        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["entities_observed"], 1)
        self.assertAlmostEqual(summary["unique_document_rate"], 2 / 3, places=6)
        self.assertEqual(by_attribute["ability"]["support_documents"], 2)
        self.assertEqual(by_attribute["hp"]["support_documents"], 2)
        self.assertEqual(by_attribute["defense"]["distortion_candidate_documents"], 1)
        self.assertEqual(by_attribute["defense"]["covered"], 0)
        self.assertEqual(by_attribute["types"]["covered"], 1)
        self.assertEqual(summary["by_attribute"]["ability"]["facts_covered"], 1)
        self.assertEqual(summary["by_attribute"]["defense"]["facts_covered"], 0)
        self.assertEqual(len(summary["_grammar_sample"]), 3)
        self.assertEqual(summary["document_words"]["p25"], 10.5)
        self.assertEqual(summary["structural_well_formedness"]["terminal_punctuation_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
