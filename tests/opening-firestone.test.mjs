import test from 'node:test';
import assert from 'node:assert/strict';
import { OpeningFirestoneCombat, OPENING_SCOPE } from '../simulator/opening-firestone.mjs';
import { makeEntity } from '../simulator/firestone.mjs';
const engine = OpeningFirestoneCombat.fromFiles();
const tribes = ['BEAST', 'MECH', 'MURLOC', 'QUILBOAR', 'UNDEAD'];
function card(id, changes = {}) {
  const d = engine.reference.get(id);
  const native = makeEntity(d, 1);
  return { card_id: id, card_type: 'minion', attack: native.attack, health: native.health,
    max_health: native.health, golden: id === 'BG32_236', taunt: native.taunt,
    divine_shield: native.divineShield, windfury: native.windfury, reborn: native.reborn,
    enchantments: [], ...changes };
}
function snapshot(board = [], hand = [], lobby = tribes) {
  return { scope: OPENING_SCOPE, turn: 1, heroId: 'TB_BaconShop_HERO_34',
    tavernTier: 1, validTribes: lobby, board, hand, health: 60, armor: 0,
    ruleset_sha256: engine.canonicalRulesetHash, reference_cards_sha256: engine.cardsHash };
}

test('actual hand Flighty Scout summons and changes first-combat outcome', () => {
  const opponent = snapshot();
  const empty = engine.evaluate({ player: snapshot(), opponent, trials: 32, seed: 17 });
  const scout = engine.evaluate({ player: snapshot([], [card('BG32_330')]), opponent, trials: 32, seed: 17 });
  assert.equal(empty.score, 0.5);
  assert.equal(scout.score, 1);
  assert.equal(scout.actualHandIncluded, true);
  assert.equal(scout.recruitReturnState.nextTurnStateAvailable, false);
});

test('missing hand and stale snapshot cannot silently become empty contexts', () => {
  const player = snapshot(); delete player.hand;
  assert.throws(() => engine.evaluate({ player, opponent: snapshot() }), /actual hand/);
  const stale = snapshot(); stale.reference_cards_sha256 = 'incorrect';
  assert.throws(() => engine.evaluate({ player: stale, opponent: snapshot() }), /provenance/);
});

test('MECH lobby enum and both current Mech minions are represented', () => {
  const player = snapshot([card('BG29_611'), card('BG26_146')]);
  const input = engine.input({ player, opponent: snapshot(), trials: 4 });
  assert.ok(input.gameState.validTribes.includes(17));
  assert.equal(input.gameState.validTribes.includes(undefined), false);
  assert.deepEqual(input.playerBoard.board[0].races, ['MECH']);
  assert.equal(engine.evaluate({ player, opponent: snapshot(), trials: 4 }).score, 1);
});

test('all current Tier1 minions enter valid first-combat simulations', () => {
  let count = 0;
  for (const id of engine.openingMinions) {
    const race = engine.reference.get(id).races?.[0];
    const lobby = [...new Set([...(race ? [race] : []), ...tribes, 'DEMON', 'DRAGON', 'ELEMENTAL', 'NAGA', 'PIRATE'])].slice(0, 5);
    const player = snapshot([card(id)], [], lobby);
    const result = engine.evaluate({ player, opponent: snapshot([], [], lobby), trials: 4, seed: 31 });
    assert.equal(result.simulations.n, 4, id);
    assert.equal(result.score, 1, id);
    count++;
  }
  assert.equal(count, 22);
});

test('current deathrattle and Rally token closures complete real combat', () => {
  for (const id of ['BG28_300', 'BG29_611', 'BG31_803', 'BG36_200', 'BG33_886']) {
    const player = snapshot([card(id)]);
    const opponent = snapshot([card('BG36_345', { attack: 4, health: 4 })]);
    const result = engine.evaluate({ player, opponent, trials: 64, seed: 121 });
    assert.equal(result.simulations.n, 64, id);
    assert.ok(Number.isFinite(result.score), id);
  }
});

test('current permanent/temporary buff provenance survives bridge without double stats', () => {
  const buffed = card('BG36_345', { attack: 5, health: 4,
    enchantments: [{ attack: 1, health: 1, blood_gem: true, source_id: 'BG20_GEM', temporary: false },
      { attack: 1, health: 0, source_id: 'BG23_000t', temporary: true }] });
  const entity = engine.input({ player: snapshot([buffed]), opponent: snapshot(), trials: 1 }).playerBoard.board[0];
  assert.equal(entity.attack, 5);
  assert.equal(entity.health, 4);
  assert.equal(entity.enchantments[0].cardId, 'BG20_GEMe');
  assert.equal(entity.enchantments[1].temporary, true);
});
