"""Lossless archive packaging, hostile manifests and non-destructive restores."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_training_evidence", ROOT / "scripts/package_training_evidence.py")
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class TrainingEvidencePackageTests(unittest.TestCase):
    def run_directory(self, temporary):
        root = Path(temporary) / "run"
        root.mkdir()
        (root / "results.json").write_text("{}\n")
        return root

    def save_manifest(self, root, manifest):
        (root / package.MANIFEST_NAME).write_text(json.dumps(manifest))

    def test_round_trip_preserves_original_gzip_bytes_and_decoded_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            payload = b'{"decision":1}\r\n' * 5
            compressed = gzip.compress(payload, mtime=123456)
            original = root / "heldout_trajectories.jsonl.gz"
            original.write_bytes(compressed)
            (root / "model.json").write_bytes(b"M" * 100)
            manifest = package.pack(root, chunk_size=17)
            archive = manifest["archives"][0]
            self.assertEqual(archive["size_bytes"], len(compressed))
            self.assertEqual(archive["sha256"], hashlib.sha256(compressed).hexdigest())
            self.assertTrue(all(part["size_bytes"] <= 17 for part in archive["parts"]))
            self.assertEqual(original.read_bytes(), compressed)
            self.assertEqual((root / "model.json").read_bytes(), b"M" * 100)
            destination = Path(temp) / "restored"
            result = package.unpack(root, output_directory=destination)
            self.assertEqual(result["restored"], [original.name])
            self.assertEqual((destination / original.name).read_bytes(), compressed)
            self.assertEqual(gzip.decompress((destination / original.name).read_bytes()), payload)

    def test_threshold_empty_archives_and_models_are_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            for name, size in (("empty.gz", 0), ("small.zip", 7), ("exact.gz", 8),
                               ("large.gz", 9), ("model.json", 100)):
                (root / name).write_bytes(b"x" * size)
            manifest = package.pack(root, chunk_size=8)
            self.assertEqual([a["file"] for a in manifest["archives"]], ["large.gz"])
            self.assertEqual([p["size_bytes"] for p in manifest["archives"][0]["parts"]], [8, 1])
            self.assertEqual((root / "empty.gz").stat().st_size, 0)
            self.assertEqual(package.DEFAULT_CHUNK_BYTES, 8 * 1024 * 1024)

    def test_pack_requires_completed_run_and_keeps_matching_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            with self.assertRaisesRegex(ValueError, "completed"):
                package.pack(root, chunk_size=8)
            self.assertFalse((root / package.MANIFEST_NAME).exists())
            (root / "results.json").write_text("{}")
            manifest = package.pack(root, chunk_size=8)
            part = root / manifest["archives"][0]["parts"][0]["file"]
            before = (part.stat().st_mtime_ns, part.stat().st_ino)
            self.assertEqual(package.pack(root, chunk_size=8), manifest)
            self.assertEqual((part.stat().st_mtime_ns, part.stat().st_ino), before)
            self.assertEqual(package.unpack(root)["reused"], ["trace.gz"])

    def test_conflicting_part_manifest_or_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            manifest = package.pack(root, chunk_size=8)
            original_manifest = (root / package.MANIFEST_NAME).read_bytes()
            part = root / manifest["archives"][0]["parts"][0]["file"]
            good_part = part.read_bytes()
            part.write_bytes(b"conflict")
            with self.assertRaisesRegex(ValueError, "overwrite unmatched"):
                package.pack(root, chunk_size=8)
            self.assertEqual(part.read_bytes(), b"conflict")
            self.assertEqual((root / package.MANIFEST_NAME).read_bytes(), original_manifest)
            part.write_bytes(good_part)
            with self.assertRaisesRegex(ValueError, "overwrite unmatched"):
                package.pack(root, chunk_size=7)
            destination = Path(temp) / "target"
            destination.mkdir()
            (destination / "trace.gz").write_bytes(b"keep this")
            with self.assertRaisesRegex(ValueError, "overwrite unmatched"):
                package.unpack(root, output_directory=destination)
            self.assertEqual((destination / "trace.gz").read_bytes(), b"keep this")

    def test_missing_corrupt_truncated_and_appended_parts_publish_no_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "a.gz").write_bytes(b"abcdefghijk")
            (root / "b.gz").write_bytes(b"lmnopqrstuv")
            manifest = package.pack(root, chunk_size=8)
            damaged = root / manifest["archives"][-1]["parts"][-1]["file"]
            good = damaged.read_bytes()
            destination = Path(temp) / "out"
            for payload in (None, b"bad", good[:-1], good + b"!"):
                with self.subTest(payload=payload):
                    if payload is None:
                        damaged.unlink()
                    else:
                        damaged.write_bytes(payload)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        package.unpack(root, output_directory=destination)
                    self.assertFalse((destination / "a.gz").exists())
                    self.assertFalse((destination / "b.gz").exists())
                    damaged.write_bytes(good)

    def test_final_whole_archive_hash_is_checked_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            manifest = package.pack(root, chunk_size=8)
            manifest["archives"][0]["sha256"] = "0" * 64
            self.save_manifest(root, manifest)
            destination = Path(temp) / "out"
            with self.assertRaisesRegex(ValueError, "Reconstructed archive"):
                package.unpack(root, output_directory=destination)
            self.assertFalse((destination / "trace.gz").exists())

    def test_traversal_absolute_windows_and_noncanonical_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            valid = package.pack(root, chunk_size=8)
            for name in ("../escape.gz", "/tmp/escape.gz", "C:\\escape.gz", "./trace.gz",
                         "notes:payload.gz", "CON.gz", "NUL.zip", "trace.gz ", "trace\n.gz"):
                bad = deepcopy(valid)
                bad["archives"][0]["file"] = name
                self.save_manifest(root, bad)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    package.unpack(root)
            for name in ("../outside", "/tmp/outside", "evidence_parts/../outside", "evidence_parts\\outside"):
                bad = deepcopy(valid)
                bad["archives"][0]["parts"][0]["file"] = name
                self.save_manifest(root, bad)
                with self.subTest(part=name), self.assertRaises(ValueError):
                    package.unpack(root)

    def test_duplicate_keys_archives_parts_and_reordered_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            valid = package.pack(root, chunk_size=8)
            variants = []
            bad = deepcopy(valid)
            bad["archives"].append(deepcopy(bad["archives"][0]))
            variants.append(bad)
            bad = deepcopy(valid)
            bad["archives"][0]["parts"][1]["file"] = bad["archives"][0]["parts"][0]["file"]
            variants.append(bad)
            bad = deepcopy(valid)
            bad["archives"][0]["parts"].reverse()
            variants.append(bad)
            bad = deepcopy(valid)
            bad["archives"][0]["size_bytes"] += 1
            variants.append(bad)
            for bad in variants:
                self.save_manifest(root, bad)
                with self.assertRaises(ValueError):
                    package.unpack(root)
            (root / package.MANIFEST_NAME).write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, "Duplicate manifest key"):
                package.unpack(root)

    def test_symlinked_original_marker_parts_manifest_and_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            outside = Path(temp) / "outside"
            outside.write_bytes(b"abcdefghijk")
            source = root / "trace.gz"
            source.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.pack(root, chunk_size=8)
            source.unlink()
            source.write_bytes(outside.read_bytes())
            marker = root / "results.json"
            marker.unlink()
            marker.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.pack(root, chunk_size=8)
            marker.unlink()
            marker.write_text("{}")
            manifest = package.pack(root, chunk_size=8)
            part = root / manifest["archives"][0]["parts"][0]["file"]
            good = part.read_bytes()
            part.unlink()
            part.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.unpack(root)
            part.unlink()
            part.write_bytes(good)
            target_dir = Path(temp) / "target"
            target_dir.mkdir()
            (target_dir / "trace.gz").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.unpack(root, output_directory=target_dir)
            manifest_path = root / package.MANIFEST_NAME
            manifest_path.unlink()
            manifest_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.unpack(root)
            self.assertEqual(outside.read_bytes(), b"abcdefghijk")

    def test_source_mutation_during_pack_is_detected_before_manifest_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            source = root / "trace.gz"
            source.write_bytes(b"abcdefghijk")
            original_matches = package._matches
            changed = False

            def mutate_after_chunk(*args):
                nonlocal changed
                if not changed:
                    changed = True
                    source.write_bytes(b"12345678901")
                return original_matches(*args)

            with patch.object(package, "_matches", side_effect=mutate_after_chunk):
                with self.assertRaisesRegex(ValueError, "changed while packaging"):
                    package.pack(root, chunk_size=8)
            self.assertFalse((root / package.MANIFEST_NAME).exists())
            self.assertFalse((root / package.PARTS_DIRECTORY).exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO conformance requires POSIX")
    def test_nonregular_inputs_are_rejected_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            os.mkfifo(root / "trace.gz")
            with self.assertRaisesRegex(ValueError, "regular file"):
                package.pack(root, chunk_size=8)

    def test_chunk_size_must_remain_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            for size in (0, -1, True, 1.5, package.DEFAULT_CHUNK_BYTES + 1):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    package.pack(root, chunk_size=size)

    def test_disappearing_competing_destination_does_not_count_as_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"abcdefghijk")
            with patch.object(package.os, "link", side_effect=FileExistsError("disappearing competitor")):
                with self.assertRaisesRegex(ValueError, "disappeared"):
                    package.pack(root, chunk_size=8)
            self.assertFalse((root / package.MANIFEST_NAME).exists())

    @unittest.skipUnless(os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"),
                         "Directory descriptor conformance requires POSIX")
    def test_swapped_parent_symlink_is_rejected_by_actual_file_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            safe = root / "parts"
            safe.mkdir()
            checked = package._path(root, "parts/archive.part")
            safe.rmdir()
            outside = Path(temp) / "outside"
            outside.mkdir()
            (outside / "archive.part").write_bytes(b"outside")
            safe.symlink_to(outside, target_is_directory=True)
            with self.assertRaises((ValueError, OSError)):
                package._open_regular(checked)

    def test_explicit_nested_archives_round_trip_and_reject_traversal_or_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_directory(temp)
            (root / "trace.gz").write_bytes(b"root archive")
            recovered = root / "recovered_expert_data"
            recovered.mkdir()
            (recovered / "trace.gz").write_bytes(b"recovered exact archive")
            second = root / "another" / "nested"
            second.mkdir(parents=True)
            (second / "other.gz").write_bytes(b"another archive")
            ignored = root / "not_selected"
            ignored.mkdir()
            (ignored / "unused.gz").write_bytes(b"leave this archive alone")
            manifest = package.pack(root, chunk_size=8,
                archive_subdirectories=["recovered_expert_data", "another/nested"])
            self.assertEqual([a["file"] for a in manifest["archives"]],
                ["another/nested/other.gz", "recovered_expert_data/trace.gz", "trace.gz"])
            target = Path(temp) / "restored"
            package.unpack(root, output_directory=target)
            for archive in manifest["archives"]:
                self.assertEqual((target / archive["file"]).read_bytes(),
                                 (root / archive["file"]).read_bytes())
            self.assertFalse((target / "not_selected").exists())
            outside = Path(temp) / "outside"
            outside.mkdir()
            (root / "linked").symlink_to(outside, target_is_directory=True)
            for directories in (["../outside"], [str(outside)], ["./recovered_expert_data"],
                                ["linked"], ["recovered_expert_data", "recovered_expert_data"]):
                with self.subTest(directories=directories), self.assertRaises(ValueError):
                    package.pack(root, chunk_size=8, archive_subdirectories=directories)
            bad = deepcopy(manifest)
            nested = bad["archives"][1]
            nested["file"] = "trace.gz/nested.gz"
            for index, part in enumerate(nested["parts"]):
                part["file"] = package._part_name(nested["file"], index)
            with self.assertRaisesRegex(ValueError, "both files and directories"):
                package.validate_manifest(bad)
            # The root completion marker remains mandatory even when the
            # selected child contains its own unrelated results.json.
            (recovered / "results.json").write_text("{}")
            (root / "results.json").unlink()
            with self.assertRaisesRegex(ValueError, "completed"):
                package.pack(root, chunk_size=8, archive_subdirectories=["recovered_expert_data"])


if __name__ == "__main__":
    unittest.main()
