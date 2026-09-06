import test from 'node:test';
import assert from 'node:assert/strict';
import { OpeningTransitionFirestone, TRANSITION_SCOPE, RECEIPT_VERSION, canonicalHash } from '../simulator/opening-transition-firestone.mjs';
import { makeEntity } from '../simulator/firestone.mjs';

const engine = OpeningTransitionFirestone.fromFiles();
const tribes = ['BEAST', 'MECH', 'MURLOC', 'QUILBOAR', 'UNDEAD'];
function card(id, changes = {}) {
  const native = makeEntity(engine.reference.get(id), 1);
  return { card_id: id, card_type: 'minion', attack: native.attack,
    health: native.health, max_health: native.health, golden: id === 'BG32_236',
    taunt: native.taunt, divine_shield: native.divineShield,
    windfury: native.windfury, reborn: native.reborn, enchantments: [], ...changes };
}
function request(turn = 1) {
  return { scope: TRANSITION_SCOPE, receipt_version: RECEIPT_VERSION,
    turn, seed: 712, pairings: [[0, 1], [2, 3], [4, 5], [6, 7]],
    snapshots: Array.from({ length: 8 }, (_, player_id) => ({
      scope: TRANSITION_SCOPE, turn, player_id, heroId: 'TB_BaconShop_HERO_34',
      tavernTier: 1, validTribes: tribes, board: [], hand: [], health: 60, armor: 0,
      ruleset_sha256: engine.canonicalRulesetHash, reference_cards_sha256: engine.cardsHash })) };
}

test('sample receipt covers eight players once and binds complete actual hands', () => {
  const input = request();
  input.snapshots[0].hand = [card('BG32_330')];
  const before = structuredClone(input);
  const receipt = engine.sample(input);
  assert.deepEqual(input, before);
  assert.deepEqual(receipt, engine.sample(input));
  assert.equal(receipt.request_sha256, canonicalHash(input));
  assert.equal(receipt.outcomes.length, 4);
  assert.ok(receipt.outcomes.every(o => o.samples === 1));
  assert.equal(receipt.outcomes[0].winner_id, 0);
  assert.equal(receipt.outcomes[0].damage_to_b, 2);
  input.snapshots[0].hand = [];
  assert.notEqual(engine.sample(input).request_sha256, receipt.request_sha256);
  assert.equal(engine.sample(input).outcomes[0].winner_id, null);
});

test('second combat uses actual turn and armor-inclusive life with current cap', () => {
  const input = request(2);
  input.snapshots[0].board = Array.from({ length: 7 }, () => card('BG36_345'));
  input.snapshots[0].tavernTier = 2;
  input.snapshots[1].armor = 3;
  const battle = engine.input({ player: input.snapshots[0], opponent: input.snapshots[1] });
  assert.equal(battle.gameState.currentTurn, 2);
  assert.equal(battle.opponentBoard.player.hpLeft, 63);
  const receipt = engine.sample(input);
  assert.equal(receipt.turn, 2);
  assert.equal(receipt.outcomes[0].uncapped_damage, 9);
  assert.equal(receipt.outcomes[0].damage_to_b, 5);
});

test('a sampled loss uses the opponent Tavern tier for exact damage', () => {
  const input = request();
  input.snapshots[1].board = [card('BG36_345')];
  input.snapshots[1].tavernTier = 3;
  const receipt = engine.sample(input);
  assert.equal(receipt.outcomes[0].winner_id, 1);
  assert.equal(receipt.outcomes[0].damage_to_a, 4);
  assert.equal(receipt.outcomes[0].damage_to_b, 0);
});

test('non-integer seeds, stale cards, mixed turns and repeated pair players abort', () => {
  for (const mutate of [
    r => { r.seed = 1.5; },
    r => { r.snapshots[0].reference_cards_sha256 = 'stale'; },
    r => { r.snapshots[0].turn = 2; },
    r => { r.pairings[0] = [0, 0]; },
    r => { r.snapshots[7].validTribes = ['DEMON', 'DRAGON', 'ELEMENTAL', 'NAGA', 'PIRATE']; },
    r => { delete r.snapshots[0].hand; },
    r => { r.snapshots[0].health = 0; },
  ]) {
    const input = request(); mutate(input);
    assert.throws(() => engine.sample(input));
  }
});

test('recruitment-eligible current pools still reject generated ordinary goldens', () => {
  const input = request();
  input.snapshots[0].board = [card('BG36_345', { golden: true })];
  assert.throws(() => engine.sample(input), /naturally golden/);
});

test('combat-only growth and deaths do not mutate recruit snapshots', () => {
  for (const id of ['BG33_886', 'BG25_013', 'BG28_300', 'BG36_200']) {
    const input = request();
    input.snapshots[0].board = [card(id)];
    input.snapshots[1].board = [card('BG36_345', { attack: 20, health: 20, max_health: 20 })];
    const before = structuredClone(input);
    const receipt = engine.sample(input);
    assert.deepEqual(input, before, id);
    assert.equal(receipt.outcomes[0].samples, 1);
  }
});
