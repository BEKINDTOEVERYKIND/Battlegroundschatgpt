"""Semantic versions of frozen positioning labels, separate from card patches."""
LEGACY_ADAPTER_VERSION = "legacy-v1"
CURRENT_ADAPTER_VERSION = "firestone-combat-v2-tribes"
ADAPTER_VERSIONS = (LEGACY_ADAPTER_VERSION, CURRENT_ADAPTER_VERSION)


def adapter_version_for_metadata(metadata: dict) -> str:
    version = metadata.get("adapterVersion", LEGACY_ADAPTER_VERSION)
    if version not in ADAPTER_VERSIONS:
        raise ValueError(f"Unsupported combat adapter version: {version}")
    schema = metadata.get("datasetSchemaVersion", metadata.get("schemaVersion", 1))
    if type(schema) is not int or schema != (2 if version == CURRENT_ADAPTER_VERSION else 1):
        raise ValueError("Combat adapter and dataset schema version mismatch")
    return version
