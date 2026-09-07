import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { PermanentCombatPrototype, PERMANENT_SCOPE, PERMANENT_RECEIPT_VERSION } from '../simulator/permanent-combat-prototype.mjs';
import { makeEntity } from '../simulator/firestone.mjs';

const require = createRequire(import.meta.url);
const stats = require('@firestone-hs/simulate-bgs-battle/dist/simulation/stats.js');
const windfury = require('@firestone-hs/simulate-bgs-battle/dist/keywords/windfury.js');
const { Simulator } = require('@firestone-hs/simulate-bgs-battle/dist/simulation/simulator.js');
const engine = PermanentCombatPrototype.fromFiles();
const tribes = ['DRAGON', 'UNDEAD', 'BEAST', 'DEMON', 'NAGA'];

function card(card_id, entity_id, changes = {}) {
  const native = makeEntity(engine.reference.get(card_id), 1);
  return { card_id, entity_id, attack: native.attack, health: native.health, max_health: native.health,
    golden: card_id.endsWith('_G'), taunt: !!native.taunt, divine_shield: !!native.divineShield,
    windfury: !!native.windfury, reborn: !!native.reborn, enchantments: [], ...changes };
}
function request(playerBoard = [], opponentBoard = []) {
  const snapshot = (player_id, board) => ({ scope: PERMANENT_SCOPE, turn: 2, player_id,
    heroId: 'TB_BaconShop_HERO_34', health: 60, armor: 0, tavernTier: 2,
    validTribes: tribes, board, hand: [], global_info: { EternalKnightsDeadThisGame: 0 },
    ruleset_sha256: engine.canonicalRulesetHash, reference_cards_sha256: engine.cardsHash });
  return { scope: PERMANENT_SCOPE, receipt_version: PERMANENT_RECEIPT_VERSION, seed: 12345,
    players_alive: 8, player: snapshot(0, playerBoard), opponent: snapshot(1, opponentBoard) };
}
const giant = () => card('BG36_345', 'enemy-giant', { attack: 100, health: 100, max_health: 100 });

test('dead original Tarecgosa keeps exact gains; surviving combat board cannot replace recruitment', () => {
  const input = request([card('BG21_015', 'tarecgosa'), card('BG26_963', 'synth')], [giant()]);
  const before = structuredClone(input);
  const oldStats = stats.modifyStats, oldWindfury = windfury.updateWindfury;
  const receipt = engine.sample(input);
  assert.deepEqual(input, before);
  assert.equal(stats.modifyStats, oldStats);
  assert.equal(windfury.updateWindfury, oldWindfury);
  assert.equal(receipt.samples, 1);
  assert.equal(receipt.full_tier2_ready, false);
  const own = receipt.writeback[0];
  assert.equal(own.upstream_public_global_counter_present, false);
  assert.deepEqual(own.surviving_original_entity_ids, []);
  assert.equal(own.entity_updates.find(e => e.entity_id === 'tarecgosa').attack_gain, 1);
  assert.equal(own.entity_updates.find(e => e.entity_id === 'tarecgosa').health_gain, 1);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.deepEqual(applied.player.board.map(c => c.entity_id), ['tarecgosa', 'synth']);
  assert.equal(applied.player.board[0].attack, 5);
  assert.equal(applied.player.board[0].health, 5);
  assert.equal(applied.player.board[1].attack, 3);
  assert.equal(applied.next_recruit_transition_implemented, false);
  assert.deepEqual(receipt, engine.sample(input));
});

test('golden Tarecgosa doubles actual grants once and preserves granted Windfury after death', () => {
  const input = request([card('BG21_015_G', 'golden-tarecgosa'), card('BG26_963_G', 'golden-synth'),
    card('BG29_810', 'paper')], [giant()]);
  const receipt = engine.sample(input);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.equal(applied.player.board[0].attack, 14);
  assert.equal(applied.player.board[0].health, 16);
  assert.equal(applied.player.board[0].windfury, true);
  assert.equal(receipt.permanent_events.filter(e => e.entity_id === 'golden-tarecgosa' && e.kind === 'stat_gain').length, 2);
  // Synthesizer also buffs Paper Drake in combat. That ordinary buff expires.
  assert.equal(applied.player.board[2].attack, 2);
  assert.equal(applied.player.board[2].health, 3);
});

test('existing Tarecgosa stats and existing keyword are not counted as new gains', () => {
  const input = request([card('BG21_015', 'tarecgosa', { attack: 9, health: 12, max_health: 12, windfury: true }),
    card('BG29_810', 'paper')], [giant()]);
  const receipt = engine.sample(input);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.equal(applied.player.board[0].attack, 10);
  assert.equal(applied.player.board[0].health, 14);
  assert.equal(applied.player.board[0].windfury, true);
  assert.equal(receipt.permanent_events.filter(e => e.kind === 'keyword_gain').length, 0);
});

test('Eternal Knight original plus Reborn death increments exact counter twice and original hand gains', () => {
  const input = request([card('BG25_008', 'knight', { reborn: true })], [giant()]);
  input.player.hand = [card('BG25_008', 'hand-knight'), card('BG25_008_G', 'golden-hand-knight')];
  const receipt = engine.sample(input);
  const own = receipt.writeback[0];
  assert.deepEqual(own.global_updates, [{ name: 'EternalKnightsDeadThisGame', before: 0, after: 2, delta: 2 }]);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.equal(applied.player.board.length, 1);
  assert.equal(applied.player.board[0].entity_id, 'knight');
  assert.equal(applied.player.board[0].reborn, true);
  assert.equal(applied.player.board[0].attack, 12);
  assert.equal(applied.player.board[0].health, 6);
  assert.equal(applied.player.hand[0].attack, 12);
  assert.equal(applied.player.hand[0].health, 6);
  assert.equal(applied.player.hand[1].attack, 24);
  assert.equal(applied.player.hand[1].health, 12);
  assert.equal(applied.player.global_info.EternalKnightsDeadThisGame, 2);
});

test('existing Knight aura is not applied twice and death counters are side-specific', () => {
  const input = request([card('BG25_008', 'old-knight', { attack: 16, health: 8, max_health: 8 })], [giant()]);
  input.player.global_info.EternalKnightsDeadThisGame = 3;
  input.opponent.global_info.EternalKnightsDeadThisGame = 7;
  const receipt = engine.sample(input);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.equal(applied.player.global_info.EternalKnightsDeadThisGame, 4);
  assert.equal(applied.opponent.global_info.EternalKnightsDeadThisGame, 7);
  assert.equal(applied.player.board[0].attack, 20);
  assert.equal(applied.player.board[0].health, 10);
});

test('both defeated sides retain their actual counters despite empty upstream public writeback', () => {
  const input = request([card('BG25_008', 'own-knight')], [card('BG25_008', 'opponent-knight')]);
  const receipt = engine.sample(input);
  assert.equal(receipt.outcome.winner_id, null);
  for (const side of receipt.writeback) {
    assert.equal(side.upstream_public_global_counter_present, false);
    assert.equal(side.global_updates[0].after, 1);
  }
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.equal(applied.player.board[0].attack, 8);
  assert.equal(applied.opponent.board[0].attack, 8);
});

test('scoped instrumentation restores every original function after an execution error', () => {
  const oldBattle = Simulator.prototype.simulateSingleBattle;
  const oldStats = stats.modifyStats, oldWindfury = windfury.updateWindfury;
  const failing = function () { throw new Error('injected engine failure'); };
  Simulator.prototype.simulateSingleBattle = failing;
  try {
    assert.throws(() => engine.sample(request([card('BG21_015', 'tarecgosa')], [])), /injected/);
    assert.equal(Simulator.prototype.simulateSingleBattle, failing);
    assert.equal(stats.modifyStats, oldStats);
    assert.equal(windfury.updateWindfury, oldWindfury);
  } finally { Simulator.prototype.simulateSingleBattle = oldBattle; }
  assert.equal(engine.sample(request([], [])).samples, 1);
});

test('combat-only summoned survivors never become recruited originals', () => {
  const input = request([card('BG28_300', 'bonehead')], [card('BG36_345', 'weak', { attack: 1, health: 1, max_health: 1 })]);
  const receipt = engine.sample(input);
  assert.ok(receipt.writeback[0].combat_only_survivors.length > 0);
  const applied = engine.applyPermanentWriteback(input, receipt);
  assert.deepEqual(applied.player.board.map(c => c.entity_id), ['bonehead']);
  assert.deepEqual(applied.player.board[0], input.player.board[0]);
});

test('Reborn Tarecgosa is rejected before executing combat in either owned zone', () => {
  const oldBattle = Simulator.prototype.simulateSingleBattle;
  let combatCalls = 0;
  Simulator.prototype.simulateSingleBattle = function (...args) { combatCalls++; return oldBattle.apply(this, args); };
  try {
    for (const cardId of ['BG21_015', 'BG21_015_G']) for (const zone of ['board', 'hand']) {
      const input = request([], []);
      input.player[zone] = [card(cardId, 'reborn-tarecgosa', { reborn: true })];
      assert.throws(() => engine.sample(input), /Reborn Tarecgosa copy persistence is unverified/);
    }
    assert.equal(combatCalls, 0);
  } finally { Simulator.prototype.simulateSingleBattle = oldBattle; }
});

test('allowed Reborn and deathrattle summons have fresh IDs beyond every original board/hand entity', () => {
  for (const id of ['BG25_008', 'BG28_300']) {
    const input = request([card(id, 'original', { reborn: true })],
      [card('BG36_345', 'weak', { attack: 4, health: 1, max_health: 1 })]);
    input.player.hand = [card('BG21_015', 'original-hand')];
    const receipt = engine.sample(input);
    const originalIds = receipt.writeback.flatMap(side => side.entity_updates.map(entity => entity.combat_entity_id));
    const summons = receipt.writeback.flatMap(side => side.combat_only_survivors);
    assert.ok(summons.length > 0);
    assert.equal(new Set(summons.map(s => s.combat_entity_id)).size, summons.length);
    for (const summoned of summons) assert.ok(summoned.combat_entity_id > Math.max(...originalIds));
    const applied = engine.applyPermanentWriteback(input, receipt);
    assert.deepEqual(applied.player.board.map(c => c.entity_id), ['original']);
    assert.deepEqual(applied.player.hand.map(c => c.entity_id), ['original-hand']);
  }
});

test('a replacement object reusing an original numeric ID cannot collect its permanent buffs', () => {
  const oldBattle = Simulator.prototype.simulateSingleBattle;
  const oldStats = stats.modifyStats;
  Simulator.prototype.simulateSingleBattle = function (player, opponent) {
    const counterfeit = { ...player.board[0] };
    stats.modifyStats(counterfeit, null, 1, 1, player.board, player.player, this.gameState);
    return oldBattle.call(this, player, opponent);
  };
  try {
    assert.throws(() => engine.sample(request([card('BG21_015', 'tarecgosa')], [])), /ID reused by a replacement/);
    assert.equal(stats.modifyStats, oldStats);
  } finally { Simulator.prototype.simulateSingleBattle = oldBattle; }
});

test('unknown families, missing identities/counters, stale provenance and unsupported enchantments abort', () => {
  for (const mutate of [
    r => { r.player.board = [card('BG29_300', 'winterfinner')]; },
    r => { delete r.player.board[0].entity_id; },
    r => { r.player.hand = [structuredClone(r.player.board[0])]; },
    r => { delete r.player.global_info.EternalKnightsDeadThisGame; },
    r => { r.player.global_info.UndeadAttackBonus = 1; },
    r => { r.player.trinkets = [{ card_id: 'BG32_MagicItem_417' }]; },
    r => { r.player.board[0].enchantments = [{ source_id: 'unknown' }]; },
    r => { r.player.reference_cards_sha256 = 'stale'; },
    r => { r.player.turn = 4; r.opponent.turn = 4; },
  ]) {
    const input = request([card('BG21_015', 'tarecgosa')], []);
    mutate(input);
    assert.throws(() => engine.sample(input));
  }
});

test('receipt binds original state and cannot insert an unknown combat entity', () => {
  const input = request([card('BG21_015', 'tarecgosa')], []);
  const receipt = engine.sample(input);
  const changed = structuredClone(input);
  changed.player.board[0].attack++;
  assert.throws(() => engine.applyPermanentWriteback(changed, receipt), /bind/);
  const corrupted = structuredClone(receipt);
  corrupted.writeback[0].entity_updates[0].entity_id = 'combat-summon';
  assert.throws(() => engine.applyPermanentWriteback(input, corrupted), /original entity/);
});
