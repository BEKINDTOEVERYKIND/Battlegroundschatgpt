# Exact permanent changes from one combat

`simulator/permanent-combat-prototype.mjs` provides a separate, versioned proof of combat persistence for **Tarecgosa and Eternal Knight**. It runs one actual seeded Firestone combat and applies permanent changes by original recruitment entity ID. Dead originals remain in the recruit board; surviving combat summons never enter it. Existing engines and running trainers do not import this prototype, and `full_tier2_ready` remains false.

Tarecgosa retains combat stat and bonus-keyword grants. Eternal Knight's current normal aura is +4/+2 for each friendly Knight death, and the golden aura is +8/+4. These agree with the frozen current card definitions and Blizzard's descriptions. [Tarecgosa card page](https://hearthstone.blizzard.com/battlegrounds/72062-tarecgosa), [Eternal Knight balance update](https://hearthstone.blizzard.com/en-us/news/24252014/35-2-2-patch-notes).

## What the existing simulator exposes—and loses

The installed, pinned `@firestone-hs/simulate-bgs-battle@1.1.750` exports `simulateSingleCombat`. Its result includes surviving entity IDs, hero global information and hands, unlike the aggregate `simulateBattle` result.

Inspection and actual tests uncovered a defect in that exact version: when a Solo side is defeated, `Simulator.simulateSingleBattle` advances its local hero variable to an absent teammate. The final result then exports an empty hand and empty global information for that side. This loses the Knight counter and any hand buffs precisely when the side dies. Both sides lose those exports on a mutual defeat. The prototype records this condition in `upstream_public_global_counter_present`.

The new wrapper retains references to both **actual original hero objects** before combat, and snapshots their mutated global information and hand after the same combat finishes. This preserves exact results without estimating deaths from aggregate outcomes. The relevant installed source files are checked against fixed SHA-256 hashes before this prototype loads.

Tarecgosa needs an additional event record: surviving stats cannot describe a dead minion's gains, and combat damage makes health-based inference unsafe. Synchronous wrappers observe the pinned `modifyStats` and `updateWindfury` calls for the original Tarecgosa entity IDs. Actual deltas already include the golden multiplier. Only the allowed companion effects can grant these changes in this fixture. All wrappers are restored in `finally`, including after an injected simulation error; no dependency files are changed.

## Deliberately narrow interface

`PermanentCombatPrototype.fromFiles()` validates current pool and source provenance. `sample(request)` accepts a pair of full original snapshots and returns an exact receipt bound to the request hash. `applyPermanentWriteback(request, receipt)` returns copied original snapshots with permanent changes applied. Hero damage remains a separately exported outcome; next-recruit refreshes, choices and triggers are not implemented by this helper.

Allowed base families are Tarecgosa, Eternal Knight, Electric Synthesizer, Thousandth Paper Drake, Risen Rider, Harmless Bonehead and Suspicious Prisonguard. The first four also allow their golden definitions. Only Patchwerk, turns 1–3, five specified tribes, eight living players and Tavern tiers 1–2 are accepted. Input globals must explicitly contain only the prior Knight death counter. Extra seasonal mechanisms and unresolved enchantments are rejected. These are conformance scenarios, not evidence that every such board is reachable on every permitted turn.

**Reborn on either normal or golden Tarecgosa is explicitly rejected in board and hand snapshots.** Copy-to-original persistence is not established by this prototype. Other allowed Reborn and Bonehead summons allocate fresh numeric IDs above every initial board/hand entity; the pinned allocation sources and actual summon tests verify this. The grant recorder additionally binds each original ID to its actual combat object, so a replacement object reusing that ID aborts rather than receiving the original's writeback.

Knight updates use the exact sampled counter delta and the current card's verified aura parameters. Original hand updates are independently checked against the simulator's actual mutated hand. Tarecgosa updates use the original-entity grant events. Current base stats, existing auras and combat-only buffs on other minions are not copied back again.

## Verified behavior

Thirteen focused tests cover normal/golden Tarecgosa gains and Windfury after death, preexisting stats, a Knight dying and returning through Reborn, normal/golden Knights in hand, preexisting counters, simultaneous defeat, rejection of unsupported state and Reborn Tarecgosa before combat, restoration after errors, request binding, fresh summon IDs, rejection of replacement objects reusing original IDs, and combat summons staying outside recruitment.

```bash
node --test tests/permanent-combat-prototype.test.mjs
```

Recorded example requests, receipts and applied original states are in `reports/permanent-combat-prototype.json`. Broader keyword sources, other permanent Tier-2 effects, and integration with the recruitment engine remain separate work.
