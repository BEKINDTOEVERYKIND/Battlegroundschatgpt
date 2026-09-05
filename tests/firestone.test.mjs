import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { FirestoneCombat, FirestoneCombatV2, CURRENT_ADAPTER_VERSION, LEGACY_ADAPTER_VERSION,
  adapterVersionForMetadata, normalizeTribes, makeEntity, sha256 } from '../simulator/firestone.mjs';
import { buildScenario, scoreScenario } from '../scripts/generate_positions.mjs';

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

test('V2 normalizes actual current Mechs in definitions, entities, and five-tribe lobbies', () => {
  const bytes = fs.readFileSync('data/reference_cards.json');
  const sim = FirestoneCombatV2.fromFiles();
  const hero = [...sim.heroIds].find(id => /Patchwerk/i.test(sim.reference.get(id).name ?? ''));
  const validTribes = ['MECH', 'BEAST', 'DEMON', 'DRAGON', 'PIRATE'];
  const card = sim.reference.get('BG26_146'); // Current Lullabot, single-tribe Mech.
  assert.deepEqual(card.races, ['MECH']);
  assert.deepEqual(sim.cards.getCard(card.id).races, ['MECH']);
  const board = [makeEntity({ ...card, races: ['MECHANICAL'] }, 10, 0, { adapterVersion: CURRENT_ADAPTER_VERSION })];
  assert.deepEqual(board[0].races, ['MECH']);
  const input = { board, opponent: [makeEntity(card, 100)], validTribes, heroId: hero, trials: 8, seed: 171 };
  const before = JSON.stringify(input);
  const result = sim.evaluate(input);
  assert.equal(result.simulations.n, 8);
  assert.deepEqual(sim.evaluate({ ...input, validTribes: ['MECHANICAL', ...validTribes.slice(1)] }), result);
  assert.equal(JSON.stringify(input), before);
  assert.throws(() => sim.evaluate({ ...input, validTribes: ['MECH', 'MECHANICAL', 'DEMON', 'DRAGON', 'PIRATE'] }), /five distinct/);
  assert.throws(() => sim.evaluate({ ...input, validTribes: ['NAGA', ...validTribes.slice(1)] }), /tribe absent/);
  // Duals are eligible when either tribe is present; neither is rejected.
  for (const id of ['BG_DEEP_015', 'BG36_764']) {
    const dual = makeEntity(sim.reference.get(id), 11);
    sim.validateBoard([dual], validTribes);
    sim.validateBoard([dual], [sim.reference.get(id).races[1], ...validTribes.slice(1)]);
    assert.throws(() => sim.validateBoard([dual], ['NAGA', ...validTribes.slice(1)]), /tribe absent/);
  }
  assert.deepEqual(normalizeTribes(['MECHANICAL', 'MECH', 'UNDEAD']), ['MECH', 'UNDEAD']);
  assert.throws(() => normalizeTribes(['NONEXISTENT']), /Unknown/);
  assert.equal(sha256(fs.readFileSync('data/reference_cards.json')), sha256(bytes));
});

test('dataset versions retain legacy reproduction and prevent silently reusing old labels', () => {
  const legacy = FirestoneCombat.fromFiles();
  const current = FirestoneCombatV2.fromFiles();
  assert.equal(legacy.adapterVersion, LEGACY_ADAPTER_VERSION);
  assert.equal(adapterVersionForMetadata({}), LEGACY_ADAPTER_VERSION);
  assert.throws(() => adapterVersionForMetadata({ adapterVersion: CURRENT_ADAPTER_VERSION }), /schema version/);
  assert.throws(() => adapterVersionForMetadata({ adapterVersion: 'unknown' }), /Unsupported/);
  assert.throws(() => current.assertMetadata({}), /differs/);
  const old = buildScenario(legacy, 0, 20260905, 5);
  assert.equal(old.metadata.adapterVersion, LEGACY_ADAPTER_VERSION);
  assert.throws(() => scoreScenario(current, old, 1), /differs/);
  const newIds = new Set(), oldIds = new Set();
  for (let i = 0; i < 200; i++) {
    for (const [engine, ids] of [[legacy, oldIds], [current, newIds]]) {
      const scenario = buildScenario(engine, i, 20260905, 5);
      assert.equal(scenario.metadata.adapterVersion, engine.adapterVersion);
      engine.validateBoard(scenario.candidates[0].board, scenario.metadata.validTribes);
      for (const card of [...scenario.candidates[0].board, ...scenario.opponent]) ids.add(card.cardId);
    }
  }
  assert.ok(newIds.has('BG26_146'), 'Corrected sampler must draw a single-tribe Mech');
  assert.ok(!oldIds.has('BG26_146'), 'Legacy sampler semantics must remain reproducible');
});
