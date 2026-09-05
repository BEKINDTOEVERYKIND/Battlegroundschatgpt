from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_live_ruleset import (LiveRulesetError, check_live_ruleset, compare_observation,
                               gallery_observation, known_issues_observation)


class LiveSourceGuardTests(unittest.TestCase):
    def test_gallery_ignores_images_but_rejects_gameplay_change(self):
        payload = {"cardCount": 1, "cards": [{"id": 12, "name": "Test", "attack": 2,
                   "text": "<b>Taunt</b>", "image": "old.png", "battlegrounds": {"tier": 1}}]}
        expected = gallery_observation(payload)
        payload["cards"][0]["image"] = "new.png"
        self.assertFalse(compare_observation(expected, gallery_observation(payload)))
        payload["cards"][0]["attack"] = 3
        self.assertTrue(compare_observation(expected, gallery_observation(payload)))


    def test_known_issues_tracks_first_post_edit_but_ignores_comments(self):
        payload = {"id": 153567, "title": "Known Issues", "post_stream": {"posts": [
            {"id": 1194899, "post_number": 1, "username": "Vyraneer", "updated_at": "t1", "cooked": "No new bans"},
            {"id": 2, "post_number": 2, "cooked": "User reply"}]}}
        expected = known_issues_observation(payload)
        payload["post_stream"]["posts"][1]["cooked"] = "Edited reply"
        self.assertEqual(expected, known_issues_observation(payload))
        payload["post_stream"]["posts"][0]["cooked"] = "New ban"
        self.assertTrue(compare_observation(expected, known_issues_observation(payload)))


    def test_offline_report_never_claims_current(self):
        with tempfile.TemporaryDirectory() as directory, patch("check_live_ruleset.fetch") as network:
            report = check_live_ruleset(Path(directory), allow_historical=True)
        self.assertEqual(report["status"], "historical")
        self.assertFalse(report["current"])
        self.assertFalse(report["network_checked"])
        network.assert_not_called()

    def test_unavailable_live_sources_block_training(self):
        with patch("check_live_ruleset.read_live_source", side_effect=TimeoutError("offline")):
            with self.assertRaises(LiveRulesetError) as captured:
                check_live_ruleset(Path(__file__).resolve().parents[1])
        self.assertFalse(captured.exception.report["current"])
        self.assertEqual(len(captured.exception.report["sources"]), 6)
        self.assertTrue(all(s["status"] == "unavailable" for s in captured.exception.report["sources"].values()))
