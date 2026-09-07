#!/usr/bin/env python3
"""Package completed-run archives as lossless, bounded binary chunks.

Archive bytes are opaque: this tool never decompresses, recompresses, inspects
holdout results, edits model JSON, or deletes original archives.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tempfile

DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MANIFEST_NAME = "training_evidence_manifest.json"
PARTS_DIRECTORY = "evidence_parts"
FORMAT = "raw-archive-chunks-v1"
ARCHIVE_SUFFIXES = (".gz", ".zip", ".tar", ".xz", ".bz2", ".zst")


def _chunk_size(value):
    if type(value) is not int or not 1 <= value <= DEFAULT_CHUNK_BYTES:
        raise ValueError("chunk_size must be an integer from 1 through 8 MiB")
    return value


def _relative_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(character) < 32 for character in name)
            or PureWindowsPath(name).drive):
        raise ValueError("Unsafe evidence path")
    path = PurePosixPath(name)
    if (path.is_absolute() or any(part in (".", "..") for part in path.parts)
            or path.as_posix() != name
            or any(part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
                   for part in path.parts)):
        raise ValueError("Unsafe evidence path")
    return path


def _root(directory, *, create=False):
    path = Path(directory).absolute()
    if path.is_symlink():
        raise ValueError("Evidence directory cannot be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("Evidence directory must exist")
    return path.resolve()


def _path(root, name):
    relative = _relative_name(name)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError("Evidence paths cannot contain symlinks")
        if current.exists() and index < len(relative.parts) - 1 and not current.is_dir():
            raise ValueError("Evidence parent is not a directory")
    return current


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def _parent_descriptor(path):
    """Walk parents without following symlinks where directory handles exist."""
    path = path.absolute()
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError("Evidence paths cannot contain symlinks")
        yield None  # Platforms without descriptor-relative filesystem calls.
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parent.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _open_regular(path):
    if path.is_symlink():
        raise ValueError("Evidence files cannot be symlinks")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with _parent_descriptor(path) as parent:
        descriptor = os.open(path.name if parent is not None else path, flags,
                             **({"dir_fd": parent} if parent is not None else {}))
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Evidence input must be a regular file")
    return os.fdopen(descriptor, "rb")


def _digest(path):
    hashed = hashlib.sha256()
    length = 0
    with _open_regular(path) as handle:
        before = _signature(os.fstat(handle.fileno()))
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_BYTES), b""):
            hashed.update(chunk)
            length += len(chunk)
        if before != _signature(os.fstat(handle.fileno())):
            raise ValueError("Evidence file changed while reading")
    if path.is_symlink() or before != _signature(path.stat()):
        raise ValueError("Evidence file changed while reading")
    return length, hashed.hexdigest()


def _matches(path, size, sha256):
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink():
        raise ValueError("Evidence files cannot be symlinks")
    if path.stat().st_size != size or _digest(path) != (size, sha256):
        raise ValueError("Refusing to overwrite unmatched destination: " + str(path))
    return True


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key: " + key)
        result[key] = value
    return result


def _load_manifest(path):
    with _open_regular(path) as stream:
        return json.load(stream, object_pairs_hook=_no_duplicate_keys)


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _part_name(original, index):
    return f"{PARTS_DIRECTORY}/{original}.part-{index:06d}"


def validate_manifest(manifest):
    """Validate all names, ordering and byte counts before touching outputs."""
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "format", "chunk_size_bytes", "archives"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["format"] != FORMAT or not isinstance(manifest["archives"], list)):
        raise ValueError("Unsupported evidence manifest")
    chunk_size = _chunk_size(manifest["chunk_size_bytes"])
    originals, parts = set(), set()
    for archive in manifest["archives"]:
        if not isinstance(archive, dict) or set(archive) != {"file", "size_bytes", "sha256", "parts"}:
            raise ValueError("Invalid archive record")
        name = archive["file"]
        if not _relative_name(name).parts or not name.endswith(ARCHIVE_SUFFIXES):
            raise ValueError("Original archive must have a safe relative archive path")
        if name in originals:
            raise ValueError("Duplicate original archive")
        originals.add(name)
        size = archive["size_bytes"]
        if type(size) is not int or size <= chunk_size or not _hash_string(archive["sha256"]):
            raise ValueError("Invalid original size or checksum")
        if (not isinstance(archive["parts"], list)
                or len(archive["parts"]) != (size + chunk_size - 1) // chunk_size):
            raise ValueError("Invalid part count")
        for index, part in enumerate(archive["parts"]):
            if not isinstance(part, dict) or set(part) != {"file", "size_bytes", "sha256"}:
                raise ValueError("Invalid part record")
            part_name = part["file"]
            _relative_name(part_name)
            if part_name in parts:
                raise ValueError("Duplicate archive part")
            parts.add(part_name)
            if part_name != _part_name(name, index):
                raise ValueError("Part path or ordering differs from its archive")
            expected_size = min(chunk_size, size - index * chunk_size)
            if (type(part["size_bytes"]) is not int or part["size_bytes"] != expected_size
                    or not _hash_string(part["sha256"])):
                raise ValueError("Invalid part size or checksum")
    all_files = originals | parts
    for name in all_files:
        if any(parent.as_posix() in all_files for parent in PurePosixPath(name).parents):
            raise ValueError("Evidence paths conflict as both files and directories")
    return manifest


def _publish(staged, target, size, sha256):
    """Atomically create without replacing an existing file, including races."""
    if _matches(target, size, sha256):
        return False
    try:
        with _parent_descriptor(staged) as source_parent, _parent_descriptor(target) as target_parent:
            if source_parent is not None and target_parent is not None and os.link in os.supports_dir_fd:
                os.link(staged.name, target.name, src_dir_fd=source_parent, dst_dir_fd=target_parent,
                        follow_symlinks=False)
            else:
                os.link(staged, target)
    except FileExistsError:
        if not _matches(target, size, sha256):
            raise ValueError("Destination disappeared during atomic publication")
        return False
    return True


def pack(directory, *, chunk_size=DEFAULT_CHUNK_BYTES, archive_subdirectories=()):
    chunk_size = _chunk_size(chunk_size)
    root = _root(directory)
    # Presence is the completed-run contract. The tool does not read scores.
    completed = _path(root, "results.json")
    if not completed.exists():
        raise ValueError("Only completed runs with results.json can be packaged")
    with _open_regular(completed):
        pass
    manifest_path = _path(root, MANIFEST_NAME)
    part_directory = _path(root, PARTS_DIRECTORY)
    if part_directory.exists() and not part_directory.is_dir():
        raise ValueError("Evidence parts destination is not a directory")
    scan_directories = [root]
    seen_subdirectories = set()
    for name in archive_subdirectories:
        relative = _relative_name(name)
        if not relative.parts or name in seen_subdirectories:
            raise ValueError("Archive subdirectories must be distinct nonempty relative paths")
        seen_subdirectories.add(name)
        selected = _path(root, name)
        if not selected.is_dir():
            raise ValueError("Archive subdirectory must exist")
        scan_directories.append(selected)
    originals = []
    for scan_directory in scan_directories:
        for candidate in sorted(scan_directory.iterdir(), key=lambda path: path.name):
            if not candidate.name.endswith(ARCHIVE_SUFFIXES):
                continue
            candidate = _path(root, candidate.relative_to(root).as_posix())
            with _open_regular(candidate) as stream:
                if os.fstat(stream.fileno()).st_size > chunk_size:
                    originals.append(candidate)
    originals.sort(key=lambda path: path.relative_to(root).as_posix())
    manifest = {"schema_version": 1, "format": FORMAT,
                "chunk_size_bytes": chunk_size, "archives": []}
    with tempfile.TemporaryDirectory(prefix=".evidence-stage-", dir=root) as temp:
        temporary = Path(temp)
        staged_parts = []
        source_signatures = []
        for original in originals:
            whole = hashlib.sha256()
            original_name = original.relative_to(root).as_posix()
            archive = {"file": original_name, "size_bytes": 0, "sha256": "", "parts": []}
            with _open_regular(original) as stream:
                before = _signature(os.fstat(stream.fileno()))
                index = 0
                while chunk := stream.read(chunk_size):
                    name = _part_name(original_name, index)
                    target = _path(root, name)
                    part = {"file": name, "size_bytes": len(chunk),
                            "sha256": hashlib.sha256(chunk).hexdigest()}
                    staged = temporary / f"part-{len(staged_parts):08d}"
                    staged.write_bytes(chunk)
                    _matches(target, part["size_bytes"], part["sha256"])
                    staged_parts.append((staged, target, part["size_bytes"], part["sha256"]))
                    archive["parts"].append(part)
                    archive["size_bytes"] += len(chunk)
                    whole.update(chunk)
                    index += 1
                if before != _signature(os.fstat(stream.fileno())):
                    raise ValueError("Original archive changed while packaging")
            source_signatures.append((original, before))
            archive["sha256"] = whole.hexdigest()
            manifest["archives"].append(archive)
        validate_manifest(manifest)
        if manifest_path.exists():
            existing = validate_manifest(_load_manifest(manifest_path))
            if existing != manifest:
                raise ValueError("Refusing to overwrite unmatched evidence manifest")
        encoded = (json.dumps(manifest, indent=2) + "\n").encode()
        staged_manifest = temporary / "manifest.json"
        staged_manifest.write_bytes(encoded)
        # Check every source again after all archives have been staged.
        for original, before in source_signatures:
            if original.is_symlink() or _signature(original.stat()) != before:
                raise ValueError("Original archive changed while packaging")
        if staged_parts:
            part_directory.mkdir(exist_ok=True)
            _path(root, PARTS_DIRECTORY)  # Reject a replaced/symlinked parent.
        for staged, target, size, sha256 in staged_parts:
            _path(root, target.relative_to(root).as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            _path(root, target.relative_to(root).as_posix())
            _publish(staged, target, size, sha256)
        if not manifest_path.exists():
            _publish(staged_manifest, manifest_path, len(encoded), hashlib.sha256(encoded).hexdigest())
        if validate_manifest(_load_manifest(manifest_path)) != manifest:
            raise ValueError("Evidence manifest changed while packaging")
    return manifest


def unpack(directory, *, output_directory=None):
    root = _root(directory)
    manifest = validate_manifest(_load_manifest(_path(root, MANIFEST_NAME)))
    destination = _root(output_directory, create=True) if output_directory is not None else root
    # Preflight targets before any archive is reconstructed or published.
    for archive in manifest["archives"]:
        _matches(_path(destination, archive["file"]), archive["size_bytes"], archive["sha256"])
        for part in archive["parts"]:
            _path(root, part["file"])
    restored, reused = [], []
    with tempfile.TemporaryDirectory(prefix=".evidence-restore-", dir=destination) as temp:
        staged_archives = []
        for index, archive in enumerate(manifest["archives"]):
            staged = Path(temp) / f"archive-{index:08d}"
            whole, size = hashlib.sha256(), 0
            with staged.open("xb") as output:
                for part in archive["parts"]:
                    hashed, part_size = hashlib.sha256(), 0
                    with _open_regular(_path(root, part["file"])) as source:
                        before = _signature(os.fstat(source.fileno()))
                        if before[2] != part["size_bytes"]:
                            raise ValueError("Archive part size or checksum mismatch")
                        while part_size < part["size_bytes"]:
                            chunk = source.read(min(DEFAULT_CHUNK_BYTES, part["size_bytes"] - part_size))
                            if not chunk:
                                break
                            output.write(chunk)
                            whole.update(chunk)
                            hashed.update(chunk)
                            size += len(chunk)
                            part_size += len(chunk)
                        if source.read(1):
                            raise ValueError("Archive part size or checksum mismatch")
                        if before != _signature(os.fstat(source.fileno())):
                            raise ValueError("Archive part changed while restoring")
                    if part_size != part["size_bytes"] or hashed.hexdigest() != part["sha256"]:
                        raise ValueError("Archive part size or checksum mismatch")
            if size != archive["size_bytes"] or whole.hexdigest() != archive["sha256"]:
                raise ValueError("Reconstructed archive size or checksum mismatch")
            staged_archives.append((staged, archive))
        # Publish only after every chunk and complete archive passed validation.
        for staged, archive in staged_archives:
            target = _path(destination, archive["file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            _path(destination, archive["file"])
            created = _publish(staged, target, archive["size_bytes"], archive["sha256"])
            (restored if created else reused).append(archive["file"])
    return {"restored": restored, "reused": reused, "archives_verified": len(manifest["archives"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    packing.add_argument("directory", type=Path)
    packing.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_BYTES,
                         help="Bytes per chunk, maximum 8 MiB; smaller values support integrity fixtures")
    packing.add_argument("--archive-subdirectory", action="append", default=[],
                         help="Also scan this safe relative subdirectory; repeat for more directories")
    restoring = commands.add_parser("unpack")
    restoring.add_argument("directory", type=Path)
    restoring.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        manifest = pack(args.directory, chunk_size=args.chunk_size,
                        archive_subdirectories=args.archive_subdirectory)
        print(json.dumps({"manifest": str(args.directory / MANIFEST_NAME),
                          "packaged_archives": len(manifest["archives"]),
                          "originals_kept": True}))
    else:
        print(json.dumps(unpack(args.directory, output_directory=args.output_directory)))


if __name__ == "__main__":
    main()
