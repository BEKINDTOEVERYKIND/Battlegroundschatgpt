#!/usr/bin/env node
/** Freeze simulation-search choices without reading stored candidate outcome labels. */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { gunzipSync, createGunzip } from 'node:zlib';
import { FirestoneCombat, ENGINE_VERSION, sha256 } from '../simulator/firestone.mjs';
import { parseArgs } from './generate_positions.mjs';

export const SEARCH_POLICY = 'neural_proposals_plus_simulation_search';
export const SEARCH_TOP_K = 3;
export const SEARCH_TRIALS = 128;
export const SEARCH_BASE_SEED = 1592602770;
export const FINAL_EVALUATION_BASE_SEED = 982451653;
export const seedForIndex = (base, index) => (base + Math.imul(index + 1, 2654435761)) >>> 0;

export function refineScenario(engine, scenario, proposal, inputIndex, proposalHash, {
  trials = SEARCH_TRIALS, seed = SEARCH_BASE_SEED, evaluationSeed = FINAL_EVALUATION_BASE_SEED,
} = {}) {
  if (!Number.isInteger(trials) || trials < 1 || !Number.isInteger(seed) || !Number.isInteger(evaluationSeed)) throw new Error('Invalid search trials/seeds');
  if (scenario.scenario_id !== proposal.scenario_id || scenario.split !== proposal.split) throw new Error('Proposal/scenario identity mismatch');
  if (scenario.metadata.cardsSha256 !== engine.cardsHash || scenario.metadata.rulesetSha256 !== engine.rulesetHash ||
      proposal.provenance.cards_sha256 !== engine.cardsHash || proposal.provenance.ruleset_sha256 !== engine.rulesetHash) {
    throw new Error('Search snapshot differs from proposal or scenario');
  }
  const ranked = proposal.ranked_candidates;
  if (!Array.isArray(ranked) || ranked.length !== scenario.candidates.length || new Set(ranked).size !== ranked.length ||
      ranked.some(i => !Number.isInteger(i) || i < 0 || i >= scenario.candidates.length)) {
    throw new Error('ranked_candidates must be a complete permutation of candidate indices');
  }
  if (ranked[0] !== proposal.selections.model) throw new Error('Model-only selection differs from first neural proposal');
  for (const i of Object.values(proposal.selections)) if (!Number.isInteger(i) || !scenario.candidates[i]) throw new Error('Invalid frozen baseline index');
  const keep = new Set([...ranked.slice(0, SEARCH_TOP_K), proposal.selections.attack, proposal.selections.taunt_last]);
  if ([...keep].some(i => !Number.isInteger(i))) throw new Error('Both attack and taunt_last baselines are required');
  // Full neural ranking is the predetermined tie order, including heuristic proposals.
  const included = ranked.filter(i => keep.has(i));
  const searchSeed = seedForIndex(seed, inputIndex);
  const finalSeed = seedForIndex(evaluationSeed, inputIndex);
  if (searchSeed === scenario.metadata.combatSeed || searchSeed === proposal.provenance.label_combat_seed || searchSeed === finalSeed) {
    throw new Error('Search RNG must differ from label and final evaluation streams');
  }
  const outcomes = included.map(candidateIndex => ({ candidateIndex, ...engine.evaluate({
    board: scenario.candidates[candidateIndex].board, opponent: scenario.opponent,
    ...scenario.metadata, trials, seed: searchSeed,
  }) }));
  if (outcomes.some(r => !Number.isFinite(r.score))) throw new Error('Nonfinite search score');
  let chosen = outcomes[0];
  for (const outcome of outcomes.slice(1)) if (outcome.score > chosen.score) chosen = outcome;
  return {
    ...proposal,
    policy: SEARCH_POLICY,
    selections: { ...proposal.selections, model: chosen.candidateIndex },
    provenance: { ...proposal.provenance,
      evaluated_policy: SEARCH_POLICY,
      selection_rule: 'Top 3 neural proposals union attack and taunt-last; maximize independent 128-trial combat score; neural rank breaks ties',
      search_top_k: SEARCH_TOP_K, search_trials: trials, search_seed: searchSeed, search_base_seed: seed,
      search_input_index: inputIndex, final_evaluation_base_seed: evaluationSeed,
      proposal_file_sha256: proposalHash, model_only_candidate: proposal.selections.model,
      search_engine: ENGINE_VERSION,
    },
    search_outcomes: outcomes,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input || !args.proposals) throw new Error('Required: --input scenarios.jsonl[.gz] --proposals frozen_proposals.jsonl[.gz]');
  const trials = Number(args.trials ?? SEARCH_TRIALS), seed = Number(args.seed ?? SEARCH_BASE_SEED);
  const evaluationSeed = Number(args['evaluation-seed'] ?? FINAL_EVALUATION_BASE_SEED);
  const workers = Number(args.workers ?? Math.min(6, os.availableParallelism()));
  const shard = Number(args.shard ?? 0), shards = Number(args.shards ?? 1);
  if (![workers, shard, shards].every(Number.isInteger) || workers < 1 || shards < 1 || shard < 0 || shard >= shards) throw new Error('Invalid worker/shard count');
  if (trials !== SEARCH_TRIALS) throw new Error('This preregistered search policy requires exactly 128 trials');
  const readBytes = file => file.endsWith('.gz') ? gunzipSync(fs.readFileSync(file)) : fs.readFileSync(file);
  const proposalBytes = readBytes(args.proposals), proposalHash = sha256(proposalBytes);
  const proposalRows = proposalBytes.toString('utf8').split('\n').filter(Boolean).map(JSON.parse);
  const proposals = new Map(proposalRows.map(r => [r.scenario_id, r]));
  if (!proposals.size || proposals.size !== proposalRows.length) throw new Error('Need nonempty unique proposal scenario IDs');
  const out = args.out ?? 'runs/refined_selections.jsonl';
  if (fs.existsSync(out)) throw new Error('Frozen search output already exists; use a new filename');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  if (workers > 1 && shards === 1) {
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'bg-refine-'));
    try {
      const completed = await Promise.allSettled(Array.from({ length: workers }, (_, i) => new Promise((resolve, reject) => {
        const childArgs = { ...args, workers: 1, shard: i, shards: workers, out: path.join(temporary, `${i}.jsonl`) };
        const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...Object.entries(childArgs).flatMap(([k, v]) => [`--${k}`, String(v)])],
          { stdio: ['ignore', 'ignore', 'inherit'] });
        child.on('error', reject);
        child.on('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`Search shard ${i} failed: ${code ?? signal}`)));
      })));
      const failure = completed.find(r => r.status === 'rejected');
      if (failure) throw failure.reason;
      const results = [];
      for (let i = 0; i < workers; ++i) {
        for (const line of fs.readFileSync(path.join(temporary, `${i}.jsonl`), 'utf8').split('\n').filter(Boolean)) results.push(JSON.parse(line));
      }
      results.sort((a, b) => a.provenance.search_input_index - b.provenance.search_input_index);
      if (results.length !== proposals.size || new Set(results.map(r => r.scenario_id)).size !== proposals.size) throw new Error('Not every proposal matched a unique scenario');
      fs.writeFileSync(out, results.map(r => JSON.stringify(r)).join('\n') + '\n', { flag: 'wx' });
      process.stdout.write(JSON.stringify({ refined: results.length, policy: SEARCH_POLICY, trials, workers, proposalSha256: proposalHash, output: out }) + '\n');
    } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
    return;
  }
  const engine = FirestoneCombat.fromFiles(args.cards, args.ruleset);
  const inputStream = fs.createReadStream(args.input);
  const lines = createInterface({ input: args.input.endsWith('.gz') ? inputStream.pipe(createGunzip()) : inputStream, crlfDelay: Infinity });
  const fd = fs.openSync(out, 'wx');
  let count = 0, index = -1;
  try {
    for await (const line of lines) {
      if (!line.trim()) continue;
      if (++index % shards !== shard) continue;
      const scenario = JSON.parse(line), proposal = proposals.get(scenario.scenario_id);
      if (!proposal) continue;
      fs.writeSync(fd, JSON.stringify(refineScenario(engine, scenario, proposal, index, proposalHash, { trials, seed, evaluationSeed })) + '\n');
      if (++count % 25 === 0) process.stderr.write(`Search shard ${shard + 1}/${shards}: ${count} scenarios\n`);
    }
  } finally { fs.closeSync(fd); lines.close(); inputStream.destroy(); }
  if (shards === 1 && count !== proposals.size) throw new Error(`Only ${count}/${proposals.size} proposal scenarios found`);
  process.stdout.write(JSON.stringify({ refined: count, policy: SEARCH_POLICY, trials, proposalSha256: proposalHash, output: out }) + '\n');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
