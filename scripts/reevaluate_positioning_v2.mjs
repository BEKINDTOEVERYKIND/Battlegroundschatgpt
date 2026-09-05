#!/usr/bin/env node
/** Audit exact published frozen orders using corrected MECH semantics. No selection or training. */
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { FirestoneCombatV2, CURRENT_ADAPTER_VERSION, ENGINE_VERSION, REFERENCE_VERSION, sha256 } from '../simulator/firestone.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i].startsWith('--') || process.argv[i + 1] == null) throw new Error('Expected --key value');
  args[process.argv[i].slice(2)] = process.argv[i + 1];
}
const RUN = 'runs/20260905-positioning-corrected-audit';
const FILES = {
  dataset: 'runs/20260905-holdout-v2/positions.jsonl.gz',
  hybrid: 'runs/20260905-positioning-search-v1/frozen_selections.jsonl',
  model: 'runs/20260905-positioning-v2/frozen_selections.jsonl',
};
const EXPECTED = {
  dataset: 'f65da09ca38cc1c11c76f9d3df9f308193134bd9e3a2f45d1a0725b5b76146e7',
  hybrid: 'e2aa136aa9ff9f5af86bebfa5aa0f5b65d9f95bda6f9ff2f3fd450bc535c5b0c',
  model: '71520d633e3b4841e328cb30f93fc187362249815fc113c758d6a70a4f29357b',
};
const BASELINES = ['random', 'attack', 'health', 'taunt_last'];
const BASE_SEED = 32452843, TRIALS = 1024;
const read = name => {
  const raw = fs.readFileSync(FILES[name]);
  const bytes = FILES[name].endsWith('.gz') ? gunzipSync(raw) : raw;
  if (sha256(bytes) !== EXPECTED[name]) throw new Error(`Frozen ${name} artifact differs from the published version`);
  return bytes.toString('utf8').trim().split('\n');
};
const scenarioLines = read('dataset');
const scenarios = scenarioLines.map(JSON.parse);
const hybrids = new Map(read('hybrid').map(line => { const r = JSON.parse(line); return [r.scenario_id, r]; }));
const models = new Map(read('model').map(line => { const r = JSON.parse(line); return [r.scenario_id, r]; }));
if (scenarios.length !== 1000 || hybrids.size !== 1000 || models.size !== 1000) throw new Error('Expected all 1,000 exact frozen test scenarios');
const engine = FirestoneCombatV2.fromFiles();
const adapterSourceHash = sha256(fs.readFileSync('simulator/firestone.mjs'));
const evaluatorSourceHash = sha256(fs.readFileSync(fileURLToPath(import.meta.url)));
const frozen = scenarios.map((scenario, index) => {
  const hybrid = hybrids.get(scenario.scenario_id), model = models.get(scenario.scenario_id);
  if (!hybrid || !model || scenario.split !== 'test' || hybrid.split !== 'test' || model.split !== 'test') throw new Error('Frozen test ID/split mismatch');
  for (const name of BASELINES) if (hybrid.selections[name] !== model.selections[name]) throw new Error('Published baseline orders differ');
  for (const selection of [hybrid, model]) {
    if (selection.provenance.dataset_sha256 !== EXPECTED.dataset || selection.provenance.cards_sha256 !== engine.cardsHash ||
        selection.provenance.ruleset_sha256 !== engine.rulesetHash) throw new Error('Selection/source snapshot mismatch');
  }
  if (scenario.metadata.cardsSha256 !== engine.cardsHash || scenario.metadata.rulesetSha256 !== engine.rulesetHash) throw new Error('Scenario data changed');
  if (hybrid.provenance.model_only_candidate !== model.selections.model || hybrid.provenance.proposal_file_sha256 !== EXPECTED.model) throw new Error('Hybrid provenance does not match the frozen model proposal');
  const expectedSearchWinner = hybrid.search_outcomes.reduce((best, row) => row.score > best.score ? row : best).candidateIndex;
  if (expectedSearchWinner !== hybrid.selections.model) throw new Error('Frozen hybrid choice is inconsistent with its published search outcomes');
  const choices = { hybrid: hybrid.selections.model, model: model.selections.model,
    ...Object.fromEntries(BASELINES.map(name => [name, model.selections[name]])) };
  for (const selected of Object.values(choices)) if (!Number.isInteger(selected) || !scenario.candidates[selected]) throw new Error('Invalid frozen candidate index');
  const combatSeed = (BASE_SEED + Math.imul(index + 1, 2654435761)) >>> 0;
  const originalEvaluationSeed = (982451653 + Math.imul(index + 1, 2654435761)) >>> 0;
  if ([scenario.metadata.combatSeed, hybrid.provenance.search_seed, originalEvaluationSeed].includes(combatSeed)) throw new Error('Fresh audit RNG overlaps label/search/original evaluation RNG');
  const allEntities = [...scenario.candidates[0].board, ...scenario.opponent];
  if (allEntities.some(e => e.races?.includes('MECHANICAL'))) throw new Error('Unexpected alias in entity data: this audit permits lobby alias correction only');
  return { choices, combatSeed, originalEvaluationSeed };
});

fs.mkdirSync(RUN, { recursive: true });
const planPath = `${RUN}/audit_plan.json`;
if (args.worker == null) {
  const workers = Number(args.workers ?? 6);
  if (!Number.isInteger(workers) || workers < 1 || workers > 16) throw new Error('Invalid workers');
  const plan = {
    audit: 'corrected_adapter_rescore_of_published_frozen_legacy_orders',
    frozen_before_audit: true, training_or_selection_performed: false,
    adapter: engine.versionMetadata(), engine: ENGINE_VERSION, referencePackage: REFERENCE_VERSION,
    source_dataset_adapter: 'legacy-v1', source_dataset_schema: 1,
    entity_changes: 'none', lobby_correction: 'MECHANICAL -> MECH',
    expected_raw_input_sha256: EXPECTED, input_files: FILES,
    adapter_source_sha256: adapterSourceHash, evaluator_source_sha256: evaluatorSourceHash,
    cards_sha256: engine.cardsHash, ruleset_sha256: engine.rulesetHash,
    scenarios: 1000, trials_per_distinct_order: TRIALS, base_seed: BASE_SEED,
    scenario_seed_formula: '(32452843 + imul(globalInputIndex + 1, 2654435761)) >>> 0',
    seed_shared_across_compared_orders: true, policies: ['hybrid', 'model'], baselines: BASELINES,
    bootstrap: { samples: 10000, seed: 20260909, method: 'paired percentile', unit: 'scenario', confidence: 0.95 },
    promotion: 'none; audit whether the previous restricted benchmark claim survives',
  };
  if (fs.existsSync(planPath)) {
    if (JSON.stringify(JSON.parse(fs.readFileSync(planPath))) !== JSON.stringify(plan)) throw new Error('Existing audit plan differs; preserve it and use a new audit');
  } else fs.writeFileSync(planPath, JSON.stringify(plan, null, 2) + '\n');
  if (args.prepare === 'true') {
    process.stdout.write(JSON.stringify({ prepared: planPath, ...engine.versionMetadata(), adapterSourceHash }) + '\n');
    process.exit(0);
  }
  if (fs.existsSync(`${RUN}/fresh_evaluation.jsonl`)) throw new Error('Completed audit already exists; refusing overwrite');
  const children = Array.from({ length: workers }, (_, worker) => new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '--worker', String(worker), '--workers', String(workers)], { stdio: ['ignore', 'ignore', 'inherit'] });
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`Audit worker ${worker} exited ${code}`)));
  }));
  const results = await Promise.allSettled(children);
  const failed = results.find(r => r.status === 'rejected');
  if (failed) throw failed.reason;
  if (sha256(fs.readFileSync('simulator/firestone.mjs')) !== adapterSourceHash) throw new Error('Adapter source changed during audit');
  const rows = [];
  for (let i = 0; i < workers; i++) for (const line of fs.readFileSync(`${RUN}/shard-${i}.jsonl`, 'utf8').trim().split('\n')) rows.push(JSON.parse(line));
  rows.sort((a, b) => a.input_index - b.input_index);
  if (rows.length !== 1000 || rows.some((r, i) => r.input_index !== i)) throw new Error('Missing or duplicate audit rows');
  fs.writeFileSync(`${RUN}/fresh_evaluation.jsonl`, rows.map(JSON.stringify).join('\n') + '\n');
  process.stdout.write(JSON.stringify({ completed: rows.length, simulations: rows.reduce((n, r) => n + r.actual_combat_simulations, 0), output: `${RUN}/fresh_evaluation.jsonl` }) + '\n');
} else {
  const plan = JSON.parse(fs.readFileSync(planPath));
  if (plan.adapter_source_sha256 !== adapterSourceHash || plan.evaluator_source_sha256 !== evaluatorSourceHash || plan.adapter.adapterVersion !== CURRENT_ADAPTER_VERSION) throw new Error('Worker/source does not match the prepared audit');
  const worker = Number(args.worker), workers = Number(args.workers);
  const fd = fs.openSync(`${RUN}/shard-${worker}.jsonl`, 'wx');
  let completed = 0;
  try {
    for (let index = worker; index < scenarios.length; index += workers) {
      const scenario = scenarios[index], { choices, combatSeed, originalEvaluationSeed } = frozen[index];
      const sourceDigest = sha256(JSON.stringify({ candidates: scenario.candidates, opponent: scenario.opponent }));
      const canonicalTribes = engine.lobbyTribes(scenario.metadata.validTribes);
      const scores = {}, cache = new Map();
      for (const [policy, candidateIndex] of Object.entries(choices)) {
        if (!cache.has(candidateIndex)) cache.set(candidateIndex, engine.evaluate({
          ...scenario.metadata, ...engine.versionMetadata(), validTribes: canonicalTribes,
          board: scenario.candidates[candidateIndex].board, opponent: scenario.opponent,
          seed: combatSeed, trials: TRIALS,
        }));
        scores[policy] = { candidate_index: candidateIndex, ...cache.get(candidateIndex) };
      }
      if (sha256(JSON.stringify({ candidates: scenario.candidates, opponent: scenario.opponent })) !== sourceDigest) throw new Error('Frozen entities or candidates were mutated');
      const row = {
        scenario_id: scenario.scenario_id, split: 'test', input_index: index, scores,
        source_scenario_sha256: sha256(scenarioLines[index]), source_boards_sha256: sourceDigest,
        legacy_lobby_tribes: scenario.metadata.validTribes, corrected_lobby_tribes: canonicalTribes,
        metadata: { ...engine.versionMetadata(), audit_only_reinterpretation: true,
          source_dataset_adapter: 'legacy-v1', source_dataset_schema: 1,
          engine: ENGINE_VERSION, referencePackage: REFERENCE_VERSION,
          cards_sha256: engine.cardsHash, ruleset_sha256: engine.rulesetHash,
          adapter_source_sha256: adapterSourceHash, selections_sha256: { hybrid: EXPECTED.hybrid, model: EXPECTED.model },
          trials: TRIALS, combat_seed: combatSeed, original_evaluation_seed: originalEvaluationSeed,
        }, actual_combat_simulations: cache.size * TRIALS,
      };
      fs.writeSync(fd, JSON.stringify(row) + '\n');
      completed++;
      if (completed % 25 === 0) process.stderr.write(`Corrected positioning audit worker ${worker + 1}/${workers}: ${completed}\n`);
    }
  } finally { fs.closeSync(fd); }
}
