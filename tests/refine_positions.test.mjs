import test from 'node:test';
import assert from 'node:assert/strict';
import { refineScenario, SEARCH_POLICY, SEARCH_BASE_SEED, FINAL_EVALUATION_BASE_SEED, seedForIndex } from '../scripts/refine_positions.mjs';

function fixture() {
  const calls = [];
  const engine = { cardsHash: 'cards', rulesetHash: 'rules', evaluate: ({ board, seed, trials }) => {
    const id = board[0].candidate;
    calls.push({ id, seed, trials });
    return { score: [0.1, 0.8, 0.8, 0.2, 0.9, 0.7][id], simulations: { n: trials, seed } };
  } };
  const candidates = Array.from({ length: 6 }, (_, candidate) => {
    const row = { board: [{ candidate }] };
    Object.defineProperty(row, 'score', { get() { throw new Error('Stored outcome label was read'); } });
    return row;
  });
  const scenario = { scenario_id: 's', split: 'test', candidates, opponent: [],
    metadata: { cardsSha256: 'cards', rulesetSha256: 'rules', combatSeed: 123 } };
  const proposal = { scenario_id: 's', split: 'test', selections: { model: 2, random: 0, attack: 1, health: 4, taunt_last: 5 },
    ranked_candidates: [2, 0, 3, 5, 1, 4], provenance: { cards_sha256: 'cards', ruleset_sha256: 'rules', label_combat_seed: 123 } };
  return { engine, scenario, proposal, calls };
}

test('search is deterministic, uses only fixed proposals, and never reads outcome labels', () => {
  const { engine, scenario, proposal, calls } = fixture();
  const first = refineScenario(engine, scenario, proposal, 7, 'frozen-proposal-hash');
  const again = refineScenario(engine, scenario, proposal, 7, 'frozen-proposal-hash');
  assert.deepEqual(first, again);
  assert.equal(first.policy, SEARCH_POLICY);
  assert.deepEqual(calls.slice(0, 5).map(r => r.id), [2, 0, 3, 5, 1]);
  assert.equal(first.selections.model, 2); // Tie with attack1 resolves by full neural rank.
  assert.equal(first.selections.health, 4); // High hidden candidate4 was never searched.
  assert.equal(first.provenance.model_only_candidate, 2);
  assert.equal(first.provenance.proposal_file_sha256, 'frozen-proposal-hash');
  assert.ok(calls.every(c => c.trials === 128 && c.seed === seedForIndex(SEARCH_BASE_SEED, 7)));
});

test('search rejects reused label and final evaluation RNG streams', () => {
  const { engine, scenario, proposal } = fixture();
  assert.throws(() => refineScenario(engine, scenario, proposal, 7, 'hash', { seed: FINAL_EVALUATION_BASE_SEED }), /RNG must differ/);
  scenario.metadata.combatSeed = seedForIndex(SEARCH_BASE_SEED, 7);
  assert.throws(() => refineScenario(engine, scenario, proposal, 7, 'hash'), /RNG must differ/);
});
