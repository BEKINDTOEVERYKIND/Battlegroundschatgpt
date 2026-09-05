#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { FirestoneCombat, TRIBES, ENGINE_VERSION, REFERENCE_VERSION, seeded, shuffle, makeEntity, sha256 } from '../simulator/firestone.mjs';

export function parseArgs(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || argv[i + 1] == null) throw new Error('Arguments must be --key value');
    options[argv[i].slice(2)] = argv[i + 1];
  }
  return options;
}

export function buildScenario(engine, index, seed, candidates = 10) {
  const rng = seeded((seed + Math.imul(index + 1, 2654435761)) >>> 0);
  const validTribes = shuffle(TRIBES, rng).slice(0, 5);
  const tier = 2 + Math.floor(rng.next() * 5);
  const available = [...engine.shopMinionIds].filter(id => !engine.positioningEligibility(id)).map(id => engine.reference.get(id)).filter(c => c.techLevel <= tier)
    .filter(c => !c.races?.length || c.races.includes('ALL') || c.races.some(r => validTribes.includes(r)));
  if (available.length < 7) throw new Error('Insufficient legal minions in lobby');
  const size = 4 + Math.floor(rng.next() * 4);
  const makeBoard = offset => {
    const theme = validTribes[Math.floor(rng.next() * validTribes.length)];
    const tribal = available.filter(c => c.races?.includes(theme));
    const picked = [];
    for (let i = 0; i < size; ++i) {
      const legal = available.filter(c => picked.filter(p => p.cardId === c.id).length < 2);
      const legalTribal = tribal.filter(c => picked.filter(p => p.cardId === c.id).length < 2);
      const pool = rng.next() < 0.7 && legalTribal.length ? legalTribal : legal;
      const card = pool[Math.floor(rng.next() * pool.length)];
      const entity = makeEntity(card, offset + i, Math.floor(rng.next() * tier * tier));
      // Independent attack/health buffs create varied positional tradeoffs.
      entity.attack += Math.floor(rng.next() * tier * 2);
      entity.health += Math.floor(rng.next() * tier * 2);
      picked.push(entity);
    }
    return picked;
  };
  const board = makeBoard(10);
  const opponent = makeBoard(100);
  const orders = [];
  const seen = new Set();
  const add = order => {
    const key = order.map(c => c.entityId).join(',');
    if (!seen.has(key) && orders.length < candidates) { seen.add(key); orders.push(order); }
  };
  add(board);
  add([...board].sort((a, b) => b.attack - a.attack));
  add([...board].sort((a, b) => a.attack - b.attack));
  add([...board].sort((a, b) => b.health - a.health));
  add([...board].sort((a, b) => Number(a.taunt) - Number(b.taunt) || b.attack - a.attack));
  for (let tries = 0; orders.length < candidates && tries < 1000; ++tries) add(shuffle(board, rng));
  // Use a hero whose power only affects starting Health; this is a valid combat
  // context with no hero power trigger, rather than omitting a combat power.
  const hero = [...engine.heroIds].map(id => engine.reference.get(id)).find(c => /Patchwerk/i.test(c.name ?? ''));
  if (!hero) throw new Error('Active Patchwerk required for neutral positioning contexts');
  const baseSeed = Math.floor(rng.next() * 2 ** 32);
  const split = index % 10 < 8 ? 'train' : index % 10 === 8 ? 'validation' : 'test';
  return {
    scenario_id: `position-${seed}-${index}`, split, opponent,
    candidates: orders.map(order => ({ board: order })),
    metadata: { source: 'synthetic_current_pool_combat', validTribes, heroId: hero.id,
      turn: 4 + tier * 2, tavernTier: tier, combatSeed: baseSeed,
      scope: 'positioning; no trinkets, gifts, quests, hand effects, or combat hero powers',
      engine: ENGINE_VERSION, referencePackage: REFERENCE_VERSION,
      cardsSha256: engine.cardsHash, rulesetSha256: engine.rulesetHash },
  };
}

export function scoreScenario(engine, scenario, trials) {
  for (const candidate of scenario.candidates) {
    Object.assign(candidate, engine.evaluate({ board: candidate.board, opponent: scenario.opponent,
      ...scenario.metadata, trials, seed: scenario.metadata.combatSeed }));
  }
  return scenario;
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  const args = parseArgs(process.argv.slice(2));
  const count = Number(args.count ?? 1000), start = Number(args.start ?? 0), seed = Number(args.seed ?? 20260905);
  const trials = Number(args.trials ?? 128), candidates = Number(args.candidates ?? 10);
  const split = args.split ?? 'auto';
  if (!['auto', 'train', 'validation', 'test'].includes(split)) throw new Error('Invalid split');
  if (![count, start, seed, trials, candidates].every(Number.isInteger) || count < 1 || start < 0 || trials < 1 || candidates < 2 || candidates > 24) {
    throw new Error('Invalid count/start/seed/trials/candidates');
  }
  const output = args.out ?? 'runs/positions.jsonl';
  const engine = FirestoneCombat.fromFiles(args.cards, args.ruleset);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output + '.coverage.json', JSON.stringify(engine.supportReport(), null, 2) + '\n');
  const fd = fs.openSync(output, 'w');
  const startTime = Date.now();
  const failures = [];
  let written = 0, combats = 0;
  try {
    for (let index = start; index < start + count; ++index) {
      const scenario = buildScenario(engine, index, seed, candidates);
      if (split !== 'auto') scenario.split = split;
      try {
        scoreScenario(engine, scenario, trials);
        fs.writeSync(fd, JSON.stringify(scenario) + '\n');
        written++;
        combats += trials * scenario.candidates.length;
      } catch (error) {
        // Broken/unsupported effects are recorded and excluded, never converted
        // into labels or counted as successful training samples.
        failures.push({ scenario_id: scenario.scenario_id, minion_ids: [...new Set([...scenario.candidates[0].board, ...scenario.opponent].map(c => c.cardId))], error: String(error) });
        process.stderr.write(`Rejected ${scenario.scenario_id}: ${error}\n`);
      }
      if ((index - start + 1) % 25 === 0) {
        process.stderr.write(JSON.stringify({ processed: index - start + 1, written, rejected: failures.length,
          combats, elapsedSeconds: (Date.now() - startTime) / 1000 }) + '\n');
      }
    }
  } finally { fs.closeSync(fd); }
  const summary = { schemaVersion: 1, attempted: count, written, rejected: failures.length, combats, trials, candidates,
    seed, start, split, engine: ENGINE_VERSION, referencePackage: REFERENCE_VERSION,
    cardsSha256: engine.cardsHash, rulesetSha256: engine.rulesetHash,
    datasetSha256: sha256(fs.readFileSync(output)), elapsedSeconds: (Date.now() - startTime) / 1000,
    source: 'synthetic_current_pool_combat', failures };
  fs.writeFileSync(output + '.meta.json', JSON.stringify(summary, null, 2) + '\n');
  process.stdout.write(JSON.stringify(summary) + '\n');
  if (!written) process.exitCode = 1;
}
