import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { HandScarletFirestone, EFFECT_TRANSPORT_VERSION } from '../simulator/offboard-scarlet-firestone.mjs';
import { TRANSITION_SCOPE, RECEIPT_VERSION, canonicalHash } from '../simulator/opening-transition-firestone.mjs';

const engine = HandScarletFirestone.fromFiles();
const require = createRequire(import.meta.url);
const { ScarletSurvivor } = require('@firestone-hs/simulate-bgs-battle/dist/cards/impl/minion/scarlet-survivor.js');
function request(consumed = true) {
  const r = { scope: TRANSITION_SCOPE, receipt_version: RECEIPT_VERSION,
    effect_transport_version: EFFECT_TRANSPORT_VERSION, turn: 1, seed: 811,
    pairings: [[0, 1], [2, 3], [4, 5], [6, 7]],
    snapshots: Array.from({ length: 8 }, (_, player_id) => ({
      scope: TRANSITION_SCOPE, effect_transport_version: EFFECT_TRANSPORT_VERSION,
      turn: 1, player_id, heroId: 'TB_BaconShop_HERO_34', tavernTier: 1,
      validTribes: ['DRAGON', 'NAGA', 'BEAST', 'MECH', 'PIRATE'],
      board: [], hand: [], health: 60, armor: 0,
      ruleset_sha256: engine.canonicalRulesetHash, reference_cards_sha256: engine.cardsHash })) };
  r.snapshots[0].board = [{ card_id: 'BG35_814', attack: 6, health: 3, max_health: 3,
    golden: false, divine_shield: false, scarlet_trigger_consumed: consumed, enchantments: [] }];
  r.snapshots[1].board = [{ card_id: 'BG36_921', attack: 5, health: 2, max_health: 2,
    golden: false, divine_shield: false, enchantments: [] }];
  return r;
}

test('real combat preserves consumed marker and exact visible shield state', () => {
  const consumed = request(true);
  const before = structuredClone(consumed);
  const output = engine.sample(consumed);
  assert.equal(output.outcomes[0].winner_id, null);
  assert.equal(output.effect_transport_version, EFFECT_TRANSPORT_VERSION);
  assert.equal(output.request_sha256, canonicalHash(consumed));
  assert.deepEqual(consumed, before);
  const shielded = request(true);
  shielded.snapshots[0].board[0].divine_shield = true;
  assert.equal(engine.sample(shielded).outcomes[0].winner_id, 0);
});

test('the actual pinned stat-change handler cannot restore a consumed shield', () => {
  for (const consumed of [true, false]) {
    const entity = engine.entity(request(consumed).snapshots[0].board[0], 100);
    const input = { target: entity, board: [entity], hero: { trinkets: [] },
      otherHero: { trinkets: [] }, gameState: { sharedState: { currentEntityId: 1000 } } };
    ScarletSurvivor.onStatsChanged(entity, input);
    assert.equal(entity.divineShield, !consumed);
    assert.equal(entity.abiityChargesLeft, 0);
    entity.divineShield = false;
    ScarletSurvivor.onStatsChanged(entity, input);
    assert.equal(entity.divineShield, false);
  }
});

test('a hand Scarlet carries both the pinned charge spelling and done enchantment', () => {
  const card = request().snapshots[0].board[0];
  const e = engine.entity(card, 121);
  assert.equal(e.abiityChargesLeft, 0);
  assert.equal(e.enchantments.filter(e => e.cardId === 'BG35_814e').length, 1);
  assert.equal(e.divineShield, false);
});

test('missing or malformed explicit markers and stale effect versions abort', () => {
  for (const mutate of [
    r => { delete r.effect_transport_version; },
    r => { delete r.snapshots[0].effect_transport_version; },
    r => { delete r.snapshots[0].board[0].scarlet_trigger_consumed; },
    r => { r.snapshots[0].board[0].scarlet_trigger_consumed = 1; },
  ]) {
    const r = request(); mutate(r);
    assert.throws(() => engine.sample(r), /Scarlet/);
  }
});
