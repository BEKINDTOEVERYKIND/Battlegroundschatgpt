/** Narrow exact-sample persistence prototype; never an all-Tier-2 engine. */
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { OpeningFirestoneCombat } from './opening-firestone.mjs';
import { canonicalHash } from './opening-transition-firestone.mjs';
import { ENGINE_VERSION, makeEntity, sha256, CURRENT_ADAPTER_VERSION, CURRENT_TRIBES } from './firestone.mjs';

const require = createRequire(import.meta.url);
const { simulateSingleCombat } = require('@firestone-hs/simulate-bgs-battle');
const statModule = require('@firestone-hs/simulate-bgs-battle/dist/simulation/stats.js');
const windfuryModule = require('@firestone-hs/simulate-bgs-battle/dist/keywords/windfury.js');
const { Simulator } = require('@firestone-hs/simulate-bgs-battle/dist/simulation/simulator.js');
const { Race } = require('@firestone-hs/reference-data');
export const PERMANENT_SCOPE = 'current-tier2-permanent-family-prototype-v1';
export const PERMANENT_RECEIPT_VERSION = 'firestone-exact-permanent-events-v1';
const TARECGOSAS = new Set(['BG21_015', 'BG21_015_G']);
const KNIGHTS = new Set(['BG25_008', 'BG25_008_G']);
const BASES = ['BG21_015', 'BG25_008', 'BG26_963', 'BG29_810', 'BG25_001', 'BG28_300', 'BG36_345'];
export const PROTOTYPE_MINIONS = new Set([...BASES, ...BASES.slice(0, 4).map(id => `${id}_G`)]);
const GLOBAL = 'EternalKnightsDeadThisGame';
const PINNED_SOURCES = {
  'simulate-bgs-battle': 'c29a6c61cc14e0dcf7a426221c24a8e5d123cd3d547fe60262ccd58c72e81120',
  'simulation/stats': '03de1f3ef6cf40e0fdf57e3e98db06ddb844922cc9a0b872087e5c54869d66d4',
  'keywords/windfury': 'b990cf743f041ed5c747dfa268e70b4c70ca7c542a692a96a3bda390ed193a5d',
  'simulation/attack': 'ea7959fef20388d2681e9985c8a93cd0af2d30706f83e0824d9cc745591a3833',
  'simulation/simulator': '1aa51e2e726c457b093be6f8adc9cfbe48681256bcdf3f02d2fc2f890adbca79',
  'simulation/deathrattle-spawns': '2f12e102c7fae4dcdf5be1d3519b11c3c3882704e1a1961675a76d0bb0b6f77f',
  'simulation/reborn': '22be9fa67320c00ff6e1dc601adc409092907cf9b09caebff62abf75df118ab7',
  'utils': '31b7ca9e7c028b8bd143a20773d8503909f692fa6f0093e9fed0387718a36aab',
};
let recording = false;

function integer(value, minimum = 0) { return Number.isSafeInteger(value) && value >= minimum; }

function verifyDependency() {
  if (ENGINE_VERSION !== '1.1.750') throw new Error('Permanent prototype requires the exact pinned Firestone version');
  for (const [file, expected] of Object.entries(PINNED_SOURCES)) {
    const path = require.resolve(`@firestone-hs/simulate-bgs-battle/dist/${file}.js`);
    if (sha256(fs.readFileSync(path)) !== expected) throw new Error(`Changed permanent-event dependency: ${file}`);
  }
}

/** Record exact actual function effects, including a Tarecgosa that later dies.
 * Only the allowlisted companions can grant Tarecgosa stats/Windfury here.
 * No dependency file is edited. Hooks are synchronous, non-reentrant, restored
 * in finally, and only observe the original battle's existing mutations.
 */
function withPermanentEvents(originals, callback) {
  if (recording) throw new Error('Permanent-event recorder is not reentrant');
  recording = true;
  const events = [];
  const oldStats = statModule.modifyStats;
  const oldWindfury = windfuryModule.updateWindfury;
  const oldBattle = Simulator.prototype.simulateSingleBattle;
  const capturedSides = new Map();
  const originalCombatEntities = new Map();
  Simulator.prototype.simulateSingleBattle = function (playerState, opponentState) {
    // Upstream 1.1.750 exports an empty hand/globalInfo for a defeated Solo
    // side because its local hero variable advances to an absent teammate.
    // Retain references to the actual original heroes BEFORE that happens.
    const heroes = [playerState.player, opponentState.player];
    for (const entity of [...playerState.board, ...playerState.player.hand,
      ...opponentState.board, ...opponentState.player.hand]) {
      const original = originals.get(entity.entityId);
      if (!original || original.card.card_id !== entity.cardId || originalCombatEntities.has(entity.entityId)) {
        throw new Error('Original combat entity identity changed before simulation');
      }
      originalCombatEntities.set(entity.entityId, entity);
    }
    if (originalCombatEntities.size !== originals.size) throw new Error('Original combat entities missing');
    const result = oldBattle.call(this, playerState, opponentState);
    for (const hero of heroes) capturedSides.set(hero.entityId, {
      globalInfo: structuredClone(hero.globalInfo), hand: structuredClone(hero.hand) });
    return result;
  };
  statModule.modifyStats = function (...args) {
    const [entity, source] = args;
    const original = originals.get(entity?.entityId);
    if (original && originalCombatEntities.get(entity.entityId) !== entity) throw new Error('Original combat entity ID reused by a replacement');
    const record = original?.zone === 'board' && TARECGOSAS.has(original.card.card_id);
    const before = record ? { attack: entity.attack, health: entity.health } : null;
    const result = oldStats(...args);
    if (record) {
      if (entity.cardId !== original.card.card_id) throw new Error('Original Tarecgosa transformed during unsupported combat');
      const isEnchantment = args[8] ?? true;
      if (!isEnchantment) throw new Error('Unsupported Tarecgosa aura/stat operation');
      const attack = entity.attack - before.attack;
      const health = entity.health - before.health;
      if (!integer(attack) || !integer(health)) throw new Error('Unsupported negative/noninteger Tarecgosa stat change');
      if (attack || health) events.push({ player_id: original.player_id, entity_id: original.card.entity_id,
        combat_entity_id: entity.entityId, card_id: entity.cardId, kind: 'stat_gain', attack, health,
        source_card_id: source?.cardId ?? null, source_combat_entity_id: source?.entityId ?? null });
    }
    return result;
  };
  windfuryModule.updateWindfury = function (...args) {
    const [entity, value] = args;
    const original = originals.get(entity?.entityId);
    if (original && originalCombatEntities.get(entity.entityId) !== entity) throw new Error('Original combat entity ID reused by a replacement');
    const previous = !!entity.windfury;
    const result = oldWindfury(...args);
    if (original?.zone === 'board' && TARECGOSAS.has(original.card.card_id) && value && !previous && entity.windfury) {
      events.push({ player_id: original.player_id, entity_id: original.card.entity_id,
        combat_entity_id: entity.entityId, card_id: entity.cardId, kind: 'keyword_gain', keyword: 'windfury' });
    }
    return result;
  };
  try { return { result: callback(), events, capturedSides }; }
  finally {
    statModule.modifyStats = oldStats;
    windfuryModule.updateWindfury = oldWindfury;
    Simulator.prototype.simulateSingleBattle = oldBattle;
    recording = false;
  }
}

export class PermanentCombatPrototype extends OpeningFirestoneCombat {
  static fromFiles(cardsPath = 'data/reference_cards.json', rulesetPath = 'data/ruleset.json') {
    verifyDependency();
    const checked = OpeningFirestoneCombat.fromFiles(cardsPath, rulesetPath);
    const prototype = new PermanentCombatPrototype({ referenceCards: JSON.parse(fs.readFileSync(cardsPath, 'utf8')),
      ruleset: checked.ruleset, cardsHash: checked.cardsHash, rulesetHash: checked.rulesetHash });
    for (const id of BASES) if (!prototype.shopMinionIds.has(id)) throw new Error(`Prototype family rotated: ${id}`);
    // These are current authoritative parameters, not inferred combat averages.
    for (const id of KNIGHTS) {
      const card = prototype.reference.get(id);
      const multiplier = id.endsWith('_G') ? 2 : 1;
      if (card.scriptDataNum1 !== 4 * multiplier || card.scriptDataNum2 !== 2 * multiplier) {
        throw new Error('Eternal Knight permanent aura changed');
      }
    }
    return prototype;
  }

  validateSnapshot(snapshot) {
    if (!snapshot || snapshot.scope !== PERMANENT_SCOPE || !integer(snapshot.turn, 1) || snapshot.turn > 3 ||
        !integer(snapshot.player_id) || snapshot.player_id > 7 || snapshot.heroId !== 'TB_BaconShop_HERO_34') {
      throw new Error('Prototype supports explicit first-three-turn Patchwerk snapshots only');
    }
    if (snapshot.ruleset_sha256 !== this.canonicalRulesetHash || snapshot.reference_cards_sha256 !== this.cardsHash) {
      throw new Error('Stale permanent snapshot provenance');
    }
    if (!integer(snapshot.health, 1) || !integer(snapshot.armor) || !integer(snapshot.tavernTier, 1) || snapshot.tavernTier > 2) {
      throw new Error('Invalid prototype player state');
    }
    if (!Array.isArray(snapshot.validTribes) || snapshot.validTribes.length !== 5 ||
        new Set(snapshot.validTribes).size !== 5 || snapshot.validTribes.some(t => !CURRENT_TRIBES.includes(t))) {
      throw new Error('Explicit five-tribe lobby required');
    }
    if (!Array.isArray(snapshot.board) || snapshot.board.length > 7 || !Array.isArray(snapshot.hand) || snapshot.hand.length > 10) {
      throw new Error('Explicit complete original board and hand required');
    }
    if (!snapshot.global_info || Object.keys(snapshot.global_info).join(',') !== GLOBAL || !integer(snapshot.global_info[GLOBAL])) {
      throw new Error('Exact original Eternal Knight counter required; other global effects unsupported');
    }
    const identities = new Set();
    const snapshotKeys = new Set(['scope', 'turn', 'player_id', 'heroId', 'health', 'armor', 'tavernTier',
      'validTribes', 'board', 'hand', 'global_info', 'ruleset_sha256', 'reference_cards_sha256']);
    if (Object.keys(snapshot).some(key => !snapshotKeys.has(key))) throw new Error('Unsupported extra snapshot mechanism');
    const cardKeys = new Set(['card_id', 'entity_id', 'card_type', 'attack', 'health', 'max_health',
      'golden', 'taunt', 'divine_shield', 'windfury', 'reborn', 'enchantments']);
    for (const card of [...snapshot.board, ...snapshot.hand]) {
      if (Object.keys(card).some(key => !cardKeys.has(key))) throw new Error('Unsupported extra minion mechanism');
      if (!PROTOTYPE_MINIONS.has(card.card_id)) throw new Error(`Unsupported permanent prototype minion: ${card.card_id}`);
      if (typeof card.entity_id !== 'string' || !card.entity_id || identities.has(card.entity_id)) throw new Error('Unique original entity IDs required');
      identities.add(card.entity_id);
      if (!integer(card.attack) || !integer(card.health, 1) || card.max_health !== card.health ||
          !Array.isArray(card.enchantments) || card.enchantments.length) throw new Error('Prototype needs undamaged stats and no unresolved enchantments');
      if (!!card.golden !== card.card_id.endsWith('_G')) throw new Error('Golden identity mismatch');
      if (TARECGOSAS.has(card.card_id) && card.reborn) throw new Error('Reborn Tarecgosa copy persistence is unverified');
      const definition = this.reference.get(card.card_id);
      if (definition.races?.length && !definition.races.some(t => t === 'ALL' || snapshot.validTribes.includes(t))) throw new Error('Minion tribe absent from lobby');
      // The allowed effect closure only grants Windfury. Extra supplied combat
      // enchantments or seasonal mechanisms are never silently ignored.
      for (const name of ['taunt', 'divine_shield', 'windfury', 'reborn']) if (typeof card[name] !== 'boolean') throw new Error(`Explicit keyword required: ${name}`);
    }
  }

  buildPair(request) {
    if (!request || request.scope !== PERMANENT_SCOPE || request.receipt_version !== PERMANENT_RECEIPT_VERSION ||
        !integer(request.seed) || request.seed > 0xffffffff || request.players_alive !== 8) throw new Error('Invalid scoped permanent combat request');
    const { player, opponent } = request;
    this.validateSnapshot(player); this.validateSnapshot(opponent);
    if (player.player_id === opponent.player_id || player.turn !== opponent.turn ||
        [...player.validTribes].sort().join() !== [...opponent.validTribes].sort().join()) throw new Error('Distinct same-turn opponents in one lobby required');
    const originals = new Map();
    const side = (snapshot, base) => {
      const entities = (cards, zone, offset) => cards.map((card, index) => {
        const entityId = base + offset + index;
        originals.set(entityId, { player_id: snapshot.player_id, zone, card });
        return { ...makeEntity(this.reference.get(card.card_id), entityId, 0, { adapterVersion: CURRENT_ADAPTER_VERSION }),
          attack: card.attack, maxAttack: card.attack, health: card.health, maxHealth: card.health,
          taunt: card.taunt, divineShield: card.divine_shield, windfury: card.windfury, reborn: card.reborn,
          enchantments: [] };
      });
      return { board: entities(snapshot.board, 'board', 0), player: {
        cardId: snapshot.heroId, entityId: base - 1, hpLeft: snapshot.health + snapshot.armor,
        tavernTier: snapshot.tavernTier, heroPowers: [], questEntities: [], questRewards: [],
        trinkets: [], secrets: [], enchantments: [], globalInfo: { ...snapshot.global_info },
        hand: entities(snapshot.hand, 'hand', 20) } };
    };
    return { originals, input: { playerBoard: side(player, 100), opponentBoard: side(opponent, 200),
      options: { numberOfSimulations: 1, applyDamageCap: false, skipInfoLogs: true, includeOutcomeSamples: false },
      gameState: { currentTurn: player.turn, validTribes: player.validTribes.map(t => Race[t]), anomalies: [], numberOfPlayersAlive: 8 } } };
  }

  sample(request) {
    const { input, originals } = this.buildPair(request);
    const { result, events, capturedSides } = withPermanentEvents(originals, () => simulateSingleCombat(structuredClone(input),
      this.cards, this.context(request.player.validTribes), { seed: request.seed, includeSpectator: false }));
    if (!result || !['won', 'lost', 'tied'].includes(result.result) || !integer(result.damageDealt)) throw new Error('Exact single combat failed');
    const sides = [[request.player, 'player', 99], [request.opponent, 'opponent', 199]];
    const writeback = sides.map(([snapshot, prefix, heroEntityId]) => {
      const captured = capturedSides.get(heroEntityId);
      const finalInfo = captured?.globalInfo;
      const before = snapshot.global_info[GLOBAL];
      const after = finalInfo?.[GLOBAL];
      if (!integer(after) || after < before) throw new Error('Exact Eternal Knight counter missing or regressed');
      const remaining = result[`${prefix}Board`];
      const hand = captured?.hand;
      if (!Array.isArray(remaining) || !Array.isArray(hand)) throw new Error('Exact ending zones missing');
      // Hand contains the originals for this restricted closure. Validate both
      // identity and exact actual hand gains against the emitted global delta.
      const originalHand = [...originals].filter(([, o]) => o.player_id === snapshot.player_id && o.zone === 'hand');
      if (hand.length !== originalHand.length) throw new Error('Unverified hand addition/removal');
      for (const [id, original] of originalHand) {
        const actual = hand.find(c => c.entityId === id);
        if (!actual || actual.cardId !== original.card.card_id) throw new Error('Original hand identity changed');
        const definition = this.reference.get(original.card.card_id);
        const attackGain = KNIGHTS.has(actual.cardId) ? (after - before) * definition.scriptDataNum1 : 0;
        const healthGain = KNIGHTS.has(actual.cardId) ? (after - before) * definition.scriptDataNum2 : 0;
        if (actual.attack !== original.card.attack + attackGain || actual.health !== original.card.health + healthGain) {
          throw new Error('Unverified actual hand stat writeback');
        }
      }
      const entityUpdates = [...originals].filter(([, o]) => o.player_id === snapshot.player_id).map(([combatId, original]) => {
        const gains = events.filter(e => e.player_id === snapshot.player_id && e.entity_id === original.card.entity_id);
        const definition = this.reference.get(original.card.card_id);
        const knight = KNIGHTS.has(original.card.card_id);
        return { entity_id: original.card.entity_id, combat_entity_id: combatId, card_id: original.card.card_id,
          zone: original.zone, attack_gain: gains.reduce((sum, e) => sum + (e.attack ?? 0), 0) + (knight ? (after - before) * definition.scriptDataNum1 : 0),
          health_gain: gains.reduce((sum, e) => sum + (e.health ?? 0), 0) + (knight ? (after - before) * definition.scriptDataNum2 : 0),
          keyword_gains: [...new Set(gains.filter(e => e.kind === 'keyword_gain').map(e => e.keyword))],
          source: knight ? 'exact_single_combat_global_counter_and_current_aura' : 'exact_original_entity_grant_events' };
      });
      return { player_id: snapshot.player_id, global_updates: [{ name: GLOBAL, before, after, delta: after - before }],
        exact_hero_entity_id: heroEntityId,
        upstream_public_global_counter_present: integer(result[`${prefix}GlobalInfo`]?.[GLOBAL]),
        entity_updates: entityUpdates,
        surviving_original_entity_ids: remaining.filter(e => originals.has(e.entityId)).map(e => originals.get(e.entityId).card.entity_id),
        combat_only_survivors: remaining.filter(e => !originals.has(e.entityId)).map(e => ({ combat_entity_id: e.entityId, card_id: e.cardId })) };
    });
    const raw = result.damageDealt;
    return { scope: PERMANENT_SCOPE, receipt_version: PERMANENT_RECEIPT_VERSION,
      request_sha256: canonicalHash(request), seed: request.seed, samples: 1, full_tier2_ready: false,
      combat_engine: `@firestone-hs/simulate-bgs-battle@${ENGINE_VERSION}`,
      dependency_source_sha256: PINNED_SOURCES,
      outcome: { winner_id: result.result === 'won' ? request.player.player_id : result.result === 'lost' ? request.opponent.player_id : null,
        uncapped_damage: raw, damage_to_player: result.result === 'lost' ? Math.min(5, raw) : 0,
        damage_to_opponent: result.result === 'won' ? Math.min(5, raw) : 0 },
      permanent_events: events, writeback };
  }

  /** Apply only permanent changes to original recruit entities, even if dead.
   * Returned snapshots are copies; no combat survivor board replaces them.
   * Hero damage is separate, and no next-recruit hooks/refresh are invented.
   */
  applyPermanentWriteback(request, receipt) {
    this.buildPair(request);
    if (receipt?.receipt_version !== PERMANENT_RECEIPT_VERSION || receipt.scope !== PERMANENT_SCOPE ||
        receipt.request_sha256 !== canonicalHash(request) || receipt.samples !== 1 || receipt.seed !== request.seed) throw new Error('Receipt does not bind this exact original state');
    const snapshots = [request.player, request.opponent].map(snapshot => {
      const copy = structuredClone(snapshot);
      const updates = receipt.writeback?.filter(w => w.player_id === snapshot.player_id);
      if (updates?.length !== 1) throw new Error('Exactly one permanent update per player required');
      const update = updates[0];
      const counter = update.global_updates?.[0];
      if (update.global_updates?.length !== 1 || counter.name !== GLOBAL || counter.before !== copy.global_info[GLOBAL] ||
          !integer(counter.after) || !integer(counter.delta) || counter.after - counter.before !== counter.delta) throw new Error('Invalid exact global update');
      const originals = [...copy.board, ...copy.hand];
      if (update.entity_updates?.length !== originals.length || new Set(update.entity_updates.map(e => e.entity_id)).size !== originals.length) throw new Error('Incomplete/duplicated original entity updates');
      for (const entity of update.entity_updates) {
        const original = originals.find(c => c.entity_id === entity.entity_id);
        if (!original || original.card_id !== entity.card_id || !integer(entity.attack_gain) || !integer(entity.health_gain) ||
            !Array.isArray(entity.keyword_gains) || entity.keyword_gains.some(k => k !== 'windfury')) throw new Error('Invalid original entity writeback');
        original.attack += entity.attack_gain;
        original.health += entity.health_gain;
        original.max_health += entity.health_gain;
        for (const keyword of entity.keyword_gains) original[keyword] = true;
      }
      copy.global_info[GLOBAL] = counter.after;
      return copy;
    });
    return { player: snapshots[0], opponent: snapshots[1], permanent_changes_only: true,
      next_recruit_transition_implemented: false, full_tier2_ready: false };
  }
}
