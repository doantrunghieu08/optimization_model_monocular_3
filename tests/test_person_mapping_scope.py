"""parse_person_mapping reads only the pkl's own folder (no parent-folder fallback)."""
import tempfile
import unittest
from pathlib import Path

from compat import parse_person_mapping


class PersonMappingScopeTest(unittest.TestCase):
    def test_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labelled = root / "exo" / "cam05" / "seq_cam05_wham"
            labelled.mkdir(parents=True)
            (labelled / "wham_seq_cam05_person_mapping.txt").write_text("0 = aria01\n4 = aria02\n")
            shared = root / "all"
            shared.mkdir()
            # a pkl with no mapping of its own must NOT inherit cam05's labels
            self.assertEqual(parse_person_mapping(shared / "wham_output_cam01.pkl"), {})
            # single mapping in the folder, other name (e.g. wham_output_cam03.pkl + wham_output_person_mapping.txt)
            (shared / "wham_output_person_mapping.txt").write_text("2 = aria01\n")
            self.assertEqual(parse_person_mapping(shared / "wham_output_cam01.pkl"), {"2": "aria01"})
            # several mappings: the pkl's own name wins, otherwise refuse to guess
            (shared / "wham_output_cam02_person_mapping.txt").write_text("9 = aria02\n")
            self.assertEqual(parse_person_mapping(shared / "wham_output_cam02.pkl"), {"9": "aria02"})
            with self.assertRaises(ValueError):
                parse_person_mapping(shared / "wham_output_cam01.pkl")


if __name__ == "__main__":
    unittest.main()
