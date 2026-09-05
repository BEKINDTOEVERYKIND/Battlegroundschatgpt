#!/usr/bin/env python3
"""Export current hsrl2 definitions and their source/dependency audit.

No external engine is needed for a data-only export. --engine-root additionally
constructs the verified external CardDB without reading its historical database.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.hsbrsim_data import CurrentDataError, build_current_database, export_current_definitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruleset", type=Path, default=ROOT / "data/ruleset.json")
    parser.add_argument("--cards", type=Path, default=ROOT / "data/reference_cards.json")
    parser.add_argument("--client-xml", type=Path, default=ROOT / "data/source/CardDefs.Bacon.xml.gz")
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--definitions-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    kwargs = dict(ruleset_path=args.ruleset, cards_path=args.cards,
                  client_xml_path=args.client_xml)
    try:
        result = (build_current_database(args.engine_root, **kwargs) if args.engine_root
                  else export_current_definitions(**kwargs))
        for path, body in ((args.definitions_out, result.definitions),
                           (args.report_out, result.provenance)):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    except (CurrentDataError, OSError, ValueError) as exc:
        print(f"Current recruit data export failed closed: {exc}", file=sys.stderr)
        return 2
    report = result.provenance
    print(json.dumps({
        "definition_count": report["definition_count"],
        "active_counts": report["active_counts"],
        "reference_dependency_closure_count": report["reference_dependency_closure_count"],
        "missing_reachable_references": report["missing_reachable_references"],
        "engine_checkout_clean": report.get("engine_checkout_clean"),
        "full_game_ready": report["full_game_ready"],
        "blockers": report["blockers"],
    }, indent=2))
    return 0  # Successful data export; full-game readiness is a separate gate.


if __name__ == "__main__":
    raise SystemExit(main())
