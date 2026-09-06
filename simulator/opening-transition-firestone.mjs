/** One actual seeded combat sample per pair, bound to complete evaluator input. */
import fs from 'node:fs';
import readline from 'node:readline';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { OpeningFirestoneCombat, OPENING_SCOPE } from './opening-firestone.mjs';
import { sha256 } from './firestone.mjs';
const require = createRequire(import.meta.url);
const { simulateBattle, createSeededRng, withRng } = require('@firestone-hs/simulate-bgs-battle');

export const TRANSITION_SCOPE = 'current-two-turn-tier1-opening-v1';
export const RECEIPT_VERSION = 'firestone-opening-outcome-v1';
const canonical = value => Array.isArray(value) ? value.map(canonical) :
  value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])])) : value;
export const canonicalHash = value => sha256(JSON.stringify(canonical(value)));

export class OpeningTransitionFirestone extends OpeningFirestoneCombat {
  static fromFiles(cardsPath = 'data/reference_cards.json', rulesetPath = 'data/ruleset.json') {
    const checked = OpeningFirestoneCombat.fromFiles(cardsPath, rulesetPath);
    return new OpeningTransitionFirestone({ referenceCards: JSON.parse(fs.readFileSync(cardsPath, 'utf8')),
      ruleset: checked.ruleset, cardsHash: checked.cardsHash, rulesetHash: checked.rulesetHash });
  }

  validateSnapshot(snapshot) {
    if (!snapshot || snapshot.scope !== TRANSITION_SCOPE || ![1, 2].includes(snapshot.turn)) {
      throw new Error('Only current two-turn opening snapshots are supported');
    }
    if (!Number.isInteger(snapshot.player_id) || snapshot.player_id < 0 || snapshot.player_id > 7 ||
        !Number.isInteger(snapshot.health) || snapshot.health <= 0 ||
        !Number.isInteger(snapshot.armor) || snapshot.armor < 0) throw new Error('Valid living player health/armor required');
    // Reuse the unchanged pool/provenance/zone validator only. The actual
    // simulation receives snapshot.turn below, including second-turn inputs.
    super.validateSnapshot({ ...snapshot, scope: OPENING_SCOPE, turn: 1 });
  }

  input({ player, opponent, trials = 1 }) {
    const input = super.input({ player, opponent, trials });
    if (player.turn !== opponent.turn || player.player_id === opponent.player_id) throw new Error('Distinct players from the same turn required');
    input.gameState.currentTurn = player.turn;
    input.playerBoard.player.hpLeft = player.health + player.armor;
    input.opponentBoard.player.hpLeft = opponent.health + opponent.armor;
    return input;
  }

  sample(request) {
    if (!request || request.scope !== TRANSITION_SCOPE || request.receipt_version !== RECEIPT_VERSION ||
        ![1, 2].includes(request.turn) || !Number.isInteger(request.seed) ||
        request.seed < 0 || request.seed > 0xffffffff) throw new Error('Invalid opening combat request');
    const { snapshots, pairings } = request;
    if (!Array.isArray(snapshots) || snapshots.length !== 8 || snapshots.some((s, i) => s.player_id !== i || s.turn !== request.turn)) {
      throw new Error('Exactly eight ordered same-turn player snapshots required');
    }
    for (const snapshot of snapshots) this.validateSnapshot(snapshot);
    if (!Array.isArray(pairings) || pairings.length !== 4 || pairings.some(p => !Array.isArray(p) || p.length !== 2) ||
        pairings.flat().some(i => !Number.isInteger(i)) || pairings.flat().sort((a, b) => a - b).join(',') !== '0,1,2,3,4,5,6,7') {
      throw new Error('Four disjoint pairings must cover all eight players');
    }
    const lobby = [...snapshots[0].validTribes].sort().join(',');
    if (snapshots.some(s => [...s.validTribes].sort().join(',') !== lobby)) throw new Error('All eight players must share one tribe lobby');
    const outcomes = pairings.map(([a, b], index) => {
      const input = this.input({ player: snapshots[a], opponent: snapshots[b], trials: 1 });
      const seed = Number.parseInt(sha256(`${request.seed}:${index}:${a}:${b}`).slice(0, 8), 16);
      const result = withRng(createSeededRng(seed), () => {
        const iterator = simulateBattle(structuredClone(input), this.cards, this.context(snapshots[a].validTribes));
        let next = iterator.next();
        while (!next.done) next = iterator.next();
        return next.value;
      });
      if (result?.won + result?.tied + result?.lost !== 1) throw new Error('A combat must complete exactly one simulation');
      const raw = result.won ? result.damageWon : result.lost ? result.damageLost : 0;
      if (!Number.isInteger(raw) || raw < 0 || raw > 13) throw new Error('Invalid exact opening combat damage');
      // Current Solo first-three-turn cap while all eight players are alive.
      const damage = Math.min(raw, 5);
      return { player_a: a, player_b: b, samples: 1, seed,
        winner_id: result.won ? a : result.lost ? b : null,
        uncapped_damage: raw, damage_to_a: result.lost ? damage : 0,
        damage_to_b: result.won ? damage : 0 };
    });
    return { scope: TRANSITION_SCOPE, receipt_version: RECEIPT_VERSION,
      request_sha256: canonicalHash(request), turn: request.turn, seed: request.seed,
      combat_engine: '@firestone-hs/simulate-bgs-battle@1.1.750', outcomes };
  }
}

// Persistent JSONL worker: simulation failures remain explicit records.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const engine = OpeningTransitionFirestone.fromFiles();
  for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    try { process.stdout.write(`${JSON.stringify(engine.sample(JSON.parse(line)))}\n`); }
    catch (error) { process.stdout.write(`${JSON.stringify({ error: String(error.message ?? error) })}\n`); }
  }
}
