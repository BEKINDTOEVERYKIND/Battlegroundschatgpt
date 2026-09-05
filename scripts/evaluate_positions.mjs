#!/usr/bin/env node
/** Rescore model/baseline selections with independent RNG, after choices are frozen. */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { gunzipSync, createGunzip } from 'node:zlib';
import { FirestoneCombat, adapterVersionForMetadata, ENGINE_VERSION, sha256 } from '../simulator/firestone.mjs';
import { parseArgs } from './generate_positions.mjs';

const args = parseArgs(process.argv.slice(2));
if (!args.input || !args.selections) throw new Error('Required: --input scenarios.jsonl --selections selections.jsonl');
const trials = Number(args.trials ?? 1024), seed = Number(args.seed ?? 982451653);
if (!Number.isInteger(trials) || trials < 1 || !Number.isInteger(seed)) throw new Error('Invalid trials/seed');
const workers = Number(args.workers ?? Math.min(6, os.availableParallelism()));
const shard = Number(args.shard ?? 0), shards = Number(args.shards ?? 1);
if (![workers, shard, shards].every(Number.isInteger) || workers < 1 || shards < 1 || shard < 0 || shard >= shards) throw new Error('Invalid worker/shard count');
const readBytes = file => file.endsWith('.gz') ? gunzipSync(fs.readFileSync(file)) : fs.readFileSync(file);
const selectionBytes = readBytes(args.selections);
const frozenRows = new Map(selectionBytes.toString('utf8').trim().split('\n').filter(Boolean).map(line => {
  const row = JSON.parse(line); return [row.scenario_id, row];
}));
const selected = new Map([...frozenRows].map(([id, row]) => [id, row.selections]));
const out = args.out ?? 'runs/fresh_evaluation.jsonl';
fs.mkdirSync(path.dirname(out), { recursive: true });

if (workers > 1 && shards === 1) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'bg-fresh-eval-'));
  try {
    const children = Array.from({ length: workers }, (_, i) => {
      const childArgs = { ...args, workers: 1, shard: i, shards: workers, out: path.join(temporary, `${i}.jsonl`) };
      return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...Object.entries(childArgs).flatMap(([k, v]) => [`--${k}`, String(v)])],
          { stdio: ['ignore', 'ignore', 'inherit'] });
        child.on('error', reject);
        child.on('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`Evaluation shard ${i} failed: ${code ?? signal}`)));
      });
    });
    const completed = await Promise.allSettled(children);
    const failure = completed.find(r => r.status === 'rejected');
    if (failure) throw failure.reason;
    const results = [];
    for (let i = 0; i < workers; ++i) {
      for (const line of fs.readFileSync(path.join(temporary, `${i}.jsonl`), 'utf8').split('\n').filter(Boolean)) results.push(JSON.parse(line));
    }
    results.sort((a, b) => a.metadata.inputIndex - b.metadata.inputIndex);
    if (results.length !== selected.size || new Set(results.map(r => r.scenario_id)).size !== selected.size) {
      throw new Error(`Only ${results.length}/${selected.size} unique selected scenarios found`);
    }
    fs.writeFileSync(out, results.map(r => JSON.stringify(r)).join('\n') + (results.length ? '\n' : ''));
    process.stdout.write(JSON.stringify({ evaluated: results.length, trials, seed, workers,
      selectionsSha256: sha256(selectionBytes), output: out }) + '\n');
  } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
} else {
const engines = new Map();
const inputStream = fs.createReadStream(args.input);
const lines = createInterface({ input: args.input.endsWith('.gz') ? inputStream.pipe(createGunzip()) : inputStream, crlfDelay: Infinity });
const fd = fs.openSync(out, 'w');
let count = 0, index = -1;
try {
  for await (const line of lines) {
    if (!line.trim()) continue;
    ++index;
    if (index % shards !== shard) continue;
    const scenario = JSON.parse(line);
    if (!selected.has(scenario.scenario_id)) continue;
    const adapterVersion = adapterVersionForMetadata(scenario.metadata);
    if (args['adapter-version'] && args['adapter-version'] !== adapterVersion) throw new Error('Requested adapter differs from dataset generation');
    if (!engines.has(adapterVersion)) engines.set(adapterVersion, FirestoneCombat.fromFiles(args.cards, args.ruleset, { adapterVersion }));
    const engine = engines.get(adapterVersion);
    if (scenario.metadata.cardsSha256 !== engine.cardsHash || scenario.metadata.rulesetSha256 !== engine.rulesetHash) {
      throw new Error('Ruleset/card snapshot differs from generation');
    }
    const selections = selected.get(scenario.scenario_id);
    const combatSeed = (seed + Math.imul(index + 1, 2654435761)) >>> 0;
    if (combatSeed === scenario.metadata.combatSeed) throw new Error('Evaluation seed must differ from label seed');
    const provenance = frozenRows.get(scenario.scenario_id).provenance;
    if (provenance?.adapter_version != null && provenance.adapter_version !== adapterVersion) throw new Error('Selection adapter differs from dataset generation');
    if (provenance?.search_seed != null && combatSeed === provenance.search_seed) throw new Error('Evaluation seed must differ from search seed');
    const cached = new Map();
    const scores = {};
    for (const [name, candidateIndex] of Object.entries(selections)) {
      if (!Number.isInteger(candidateIndex) || !scenario.candidates[candidateIndex]) throw new Error(`Invalid selected index ${candidateIndex}`);
      if (!cached.has(candidateIndex)) cached.set(candidateIndex, engine.evaluate({
        board: scenario.candidates[candidateIndex].board, opponent: scenario.opponent,
        ...scenario.metadata, trials, seed: combatSeed,
      }));
      scores[name] = { candidateIndex, ...cached.get(candidateIndex) };
    }
    fs.writeSync(fd, JSON.stringify({ scenario_id: scenario.scenario_id, split: scenario.split, scores,
      metadata: { ...engine.versionMetadata(), trials, combatSeed, inputIndex: index, engine: ENGINE_VERSION,
        evaluatedPolicy: provenance?.evaluated_policy ?? 'neural_model_only', selectionsSha256: sha256(selectionBytes),
        cardsSha256: engine.cardsHash, rulesetSha256: engine.rulesetHash } }) + '\n');
    if (++count % 25 === 0) process.stderr.write(`Fresh evaluation shard ${shard + 1}/${shards}: ${count} scenarios\n`);
  }
} finally { fs.closeSync(fd); lines.close(); inputStream.destroy(); }
if (shards === 1 && count !== selected.size) throw new Error(`Only ${count}/${selected.size} selected scenarios found`);
process.stdout.write(JSON.stringify({ evaluated: count, trials, seed, selectionsSha256: sha256(selectionBytes), output: out }) + '\n');
}
