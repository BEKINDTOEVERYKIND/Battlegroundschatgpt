# Third-party sources

- **Firestone combat simulator**: `@firestone-hs/simulate-bgs-battle` 1.1.750,
  published by Sebastien Tromp. npm package metadata declares MIT. Imported as a
  pinned dependency; source is not copied into this repository.
  https://www.npmjs.com/package/@firestone-hs/simulate-bgs-battle
- **Firestone reference package**: `@firestone-hs/reference-data` 3.0.196,
  pinned directly and through a dependency override. Retain upstream package
  notices when redistributing dependencies.
  https://www.npmjs.com/package/@firestone-hs/reference-data
- **HSBRSIM**: Gallo13th/HSBRSIM, pinned in
  `config/recruit-engine.lock.json`. Used from a verified external checkout for
  current-data construction, explicit recruitment fixtures, and effect tests.
  `pyproject.toml` declares MIT; the inspected revision has no standalone license
  file. No source from it is vendored here.
  https://github.com/Gallo13th/HSBRSIM
- **Game data**: Hearthstone and Battlegrounds game content belongs to Blizzard
  Entertainment. Snapshot records derive from Blizzard's official card gallery
  and client definitions tracked by HearthSim/hsdata. Source locations, revisions,
  and checksums appear in `data/ruleset.json`. Game content is not relicensed by
  this project's original code. This is an independent research project.
  https://hearthstone.blizzard.com/en-us/battlegrounds
  https://github.com/HearthSim/hsdata

The installed dependency license terms continue to apply. The machine-readable
snapshots preserve provenance for offline reproducibility; reference-only card
definitions are not permission to use those cards in the active shop pool.
