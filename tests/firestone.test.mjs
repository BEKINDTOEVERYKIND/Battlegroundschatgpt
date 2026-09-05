import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { FirestoneCombat, makeEntity, sha256 } from '../simulator/firestone.mjs';

const tribes = ['BEAST', 'DEMON', 'DRAGON', 'MECHANICAL', 'MURLOC'];
const definitions = [
  { id: 'TEST_MINION', dbfId: 990001, name: 'Test minion', type: 'MINION', set: 'Battlegrounds',
    techLevel: 1, attack: 2, health: 2, races: [], mechanics: [], isBaconPool: true },
  { id: 'TEST_RETIRED', dbfId: 990002, name: 'Retired fixture', type: 'MINION', set: 'Battlegrounds',
    techLevel: 1, attack: 2, health: 2, races: [], mechanics: [], isBaconPool: true },
  { id: 'TEST_HERO', dbfId: 990003, name: 'Patchwerk', type: 'HERO', set: 'Battlegrounds', health: 30 },
];
const ruleset = { status: 'verified', active: { minion_ids: ['TEST_MINION'], hero_ids: ['TEST_HERO'] } };
const engine = () => new FirestoneCombat({ referenceCards: structuredClone(definitions), ruleset });
const entity = (id, attack, health, extra = {}) => ({ ...makeEntity(definitions[0], id), attack, health, ...extra });
const evaluate = (sim, board, opponent, seed = 1) => sim.evaluate({ board, opponent, validTribes: tribes,
  heroId: 'TEST_HERO', trials: 32, seed, tavernTier: 4 });

test('simulates known simultaneous damage and damage totals', () => {
  const sim = engine();
  const tie = evaluate(sim, [entity(10, 2, 2)], [entity(100, 2, 2)]);
  assert.equal(tie.simulations.tied, 32);
  assert.equal(tie.score, 0.5);
  const win = evaluate(sim, [entity(10, 10, 10)], [entity(100, 1, 1)]);
  assert.equal(win.simulations.won, 32);
  assert.equal(win.simulations.meanNetDamage, 5);
});

test('Divine Shield prevents lethal simultaneous damage', () => {
  const result = evaluate(engine(), [entity(10, 1, 1, { divineShield: true })], [entity(100, 1, 1)]);
  assert.equal(result.simulations.won, 32);
});

test('taunt attracts attacks and changes the outcome', () => {
  // Larger board attacks first. The 1/1 must hit the 1/100 taunt; without
  // taunt it can remove the 100/1 attacker and save the friendly 2/2.
  const sim = engine();
  const board = [entity(10, 1, 1), entity(11, 2, 2), entity(12, 0, 1)];
  const opponents = [entity(100, 100, 1), entity(101, 1, 100, { taunt: true })];
  const taunt = evaluate(sim, board, opponents, 345);
  const noTaunt = evaluate(sim, board, opponents.map(m => ({ ...m, taunt: false })), 345);
  assert.equal(taunt.simulations.lost, 32);
  assert.ok(noTaunt.simulations.meanNetDamage > taunt.simulations.meanNetDamage);
});

test('A-B-A repeated simulations are deterministic and do not mutate caller data', () => {
  const sim = engine();
  const board = [entity(10, 4, 5), entity(11, 2, 2, { divineShield: true }), entity(12, 7, 2)];
  const opponent = [entity(100, 4, 3), entity(101, 8, 2), entity(102, 1, 10, { taunt: true })];
  const before = JSON.stringify({ board, opponent });
  const a = evaluate(sim, board, opponent, 76);
  evaluate(sim, [...board].reverse(), opponent, 123);
  const again = evaluate(sim, board, opponent, 76);
  assert.deepEqual(a, again);
  assert.equal(JSON.stringify({ board, opponent }), before);
});

test('rejects retired cards, invalid lobby tribes, and missing or changed snapshots', () => {
  const sim = engine();
  assert.throws(() => evaluate(sim, [entity(10, 2, 2, { cardId: 'TEST_RETIRED' })], []), /Inactive minion/);
  assert.throws(() => sim.evaluate({ board: [], opponent: [], validTribes: [], heroId: 'TEST_HERO' }), /five distinct/);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bg-test-'));
  const cardsPath = path.join(dir, 'cards.json'), rulesPath = path.join(dir, 'rules.json');
  try {
    const bytes = JSON.stringify(definitions);
    fs.writeFileSync(cardsPath, bytes);
    const relative = path.relative(path.dirname(rulesPath), cardsPath).replaceAll(path.sep, '/');
    fs.writeFileSync(rulesPath, JSON.stringify({ ...ruleset, checksums: { [relative]: sha256(bytes) } }));
    assert.equal(FirestoneCombat.fromFiles(cardsPath, rulesPath).minionIds.size, 1);
    fs.appendFileSync(cardsPath, ' ');
    assert.throws(() => FirestoneCombat.fromFiles(cardsPath, rulesPath), /changed card snapshot/);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('current real Deathrattle combat executes and preserves input', { skip: !fs.existsSync('data/reference_cards.json') }, () => {
  const sim = FirestoneCombat.fromFiles();
  const card = sim.reference.get('BG28_300'); // Harmless Bonehead -> two 1/1 Skeletons.
  const defender = sim.reference.get('BG22_202'); // Tad's only effect is on selling.
  assert.ok(sim.minionIds.has(card.id) && sim.minionIds.has(defender.id));
  const hero = [...sim.heroIds].find(id => /Patchwerk/i.test(sim.reference.get(id).name ?? ''));
  const validTribes = ['BEAST', 'DEMON', 'MURLOC', 'PIRATE', 'UNDEAD'];
  const minion = makeEntity(card, 10);
  const opponent = makeEntity(defender, 100);
  const snapshot = JSON.stringify([minion, opponent]);
  const result = sim.evaluate({ board: [minion], opponent: [opponent], validTribes, heroId: hero, trials: 16, seed: 125 });
  assert.equal(result.simulations.n, 16);
  assert.equal(result.simulations.won, 16);
  assert.equal(result.simulations.meanNetDamage, 5);
  assert.equal(JSON.stringify([minion, opponent]), snapshot);
});
