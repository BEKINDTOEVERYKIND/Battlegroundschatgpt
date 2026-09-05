import json
import tempfile
import unittest
from pathlib import Path

from bg_ai.provenance import SnapshotError, file_sha256, load_snapshot


class SnapshotIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cards = self.root / "cards.json"
        self.cards.write_text('[{"id":"fixture"}]')
        self.path = self.root / "ruleset.json"
        self.manifest = {
            "mode": "solo", "ruleset_id": "test-fixture", "patch": "fixture", "build": 1,
            "status": "verified", "active": {"minion_ids": ["fixture"]},
            "checksums": {"cards.json": file_sha256(self.cards)},
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_verified_data_is_not_full_game_certification(self):
        load_snapshot(self.path)
        with self.assertRaisesRegex(SnapshotError, "Full-game"):
            load_snapshot(self.path, scope="full_game")

    def test_silent_data_replacement_is_rejected(self):
        self.cards.write_text('[{"id":"retired-card"}]')
        with self.assertRaisesRegex(SnapshotError, "changed"):
            load_snapshot(self.path)

    def test_unverified_data_rejected_unless_combat_explicitly_verified(self):
        self.manifest["status"] = "blocked"
        self.save()
        with self.assertRaisesRegex(SnapshotError, "unverified"):
            load_snapshot(self.path)
        self.manifest["combat_pool_verified"] = True
        self.save()
        load_snapshot(self.path)

    def test_duos_and_duplicate_cards_are_rejected(self):
        self.manifest["mode"] = "duos"
        self.save()
        with self.assertRaisesRegex(SnapshotError, "Solo"):
            load_snapshot(self.path)
        self.manifest["mode"] = "solo"
        self.manifest["active"]["minion_ids"].append("fixture")
        self.save()
        with self.assertRaisesRegex(SnapshotError, "Duplicate"):
            load_snapshot(self.path)

    def test_checksums_are_mandatory(self):
        self.manifest.pop("checksums")
        self.save()
        with self.assertRaisesRegex(SnapshotError, "checksums"):
            load_snapshot(self.path)


if __name__ == "__main__":
    unittest.main()
