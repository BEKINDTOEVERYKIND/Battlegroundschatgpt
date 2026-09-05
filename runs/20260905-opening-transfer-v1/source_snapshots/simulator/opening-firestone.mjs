/** Current complete Tier1 first-combat bridge; v1 positioning remains untouched. */
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { FirestoneCombatV2, makeEntity, sha256, CURRENT_TRIBES } from './firestone.mjs';
const require = createRequire(import.meta.url);
const { Race } = require('@firestone-hs/reference-data');
const { simulateBattle, createSeededRng, withRng } = require('@firestone-hs/simulate-bgs-battle');
export const OPENING_SCOPE = 'current-complete-tier1-opening-v1';
const canonical = value => Array.isArray(value) ? value.map(canonical) :
  value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])])) : value;
const normalize = races => (races ?? []).map(r => r === 'MECHANICAL' ? 'MECH' : r);
const TRIBES = CURRENT_TRIBES;

export class OpeningFirestoneCombat extends FirestoneCombatV2 {
  constructor(options) {
    super({ ...options, referenceCards: options.referenceCards.map(c => ({ ...c, races: normalize(c.races) })) });
    this.canonicalRulesetHash = sha256(JSON.stringify(canonical(options.ruleset)));
    this.openingMinions = new Set([...this.shopMinionIds].filter(id => this.reference.get(id).techLevel === 1));
    this.openingSpells = new Set((this.ruleset.active?.tavern_spell_ids ?? []).filter(id => this.reference.get(id).techLevel === 1));
    if (this.openingMinions.size !== 22 || this.openingSpells.size !== 8) throw new Error('Current opening pool changed; validate conformance');
  }

  static fromFiles(cardsPath = 'data/reference_cards.json', rulesetPath = 'data/ruleset.json') {
    // Let the existing frozen-source loader verify checksums and active status.
    const checked = FirestoneCombatV2.fromFiles(cardsPath, rulesetPath);
    return new OpeningFirestoneCombat({ referenceCards: JSON.parse(fs.readFileSync(cardsPath, 'utf8')),
      ruleset: checked.ruleset, cardsHash: checked.cardsHash, rulesetHash: checked.rulesetHash });
  }

  validateSnapshot(snapshot) {
    if (!snapshot || snapshot.scope !== OPENING_SCOPE || snapshot.turn !== 1) throw new Error('Only explicit current first-combat snapshots are supported');
    if (snapshot.ruleset_sha256 !== this.canonicalRulesetHash || snapshot.reference_cards_sha256 !== this.cardsHash) throw new Error('Stale opening snapshot provenance');
    if (snapshot.heroId !== 'TB_BaconShop_HERO_34') throw new Error('Opening combat supports Patchwerk only');
    if (!Number.isInteger(snapshot.tavernTier) || snapshot.tavernTier < 1 || snapshot.tavernTier > 6) throw new Error('Invalid opening Tavern tier');
    const tribes = snapshot.validTribes;
    if (!Array.isArray(tribes) || tribes.length !== 5 || new Set(tribes).size !== 5 || tribes.some(t => !TRIBES.includes(t))) throw new Error('Five distinct valid tribes required');
    if (!Array.isArray(snapshot.board) || snapshot.board.length > 7 || !Array.isArray(snapshot.hand) || snapshot.hand.length > 10) throw new Error('Complete legal-sized board AND actual hand are required');
    for (const card of [...snapshot.board, ...snapshot.hand]) {
      const id = card.card_id;
      const definition = this.reference.get(id);
      if (!definition) throw new Error(`Unknown opening definition: ${id}`);
      const isMinion = this.openingMinions.has(id);
      if (!isMinion && !this.openingSpells.has(id) && !['BG20_GEM', 'BG23_000t'].includes(id)) throw new Error(`Unsupported opening hand entity: ${id}`);
      if (snapshot.board.includes(card) && !isMinion) throw new Error('Opening battlefield may only contain current Tier1 minions');
      if (isMinion) {
        if (!Number.isFinite(card.attack) || card.attack < 0 || !Number.isFinite(card.health) || card.health <= 0) throw new Error('Invalid opening minion stats');
        if (definition.races?.length && !definition.races.some(r => r === 'ALL' || tribes.includes(r))) throw new Error('Opening minion tribe absent from lobby');
        if (!!card.golden !== (id === 'BG32_236')) throw new Error('Only naturally golden Aureate is permitted');
      }
    }
  }

  entity(card, entityId) {
    const definition = this.reference.get(card.card_id);
    const entity = makeEntity(definition, entityId);
    if (this.openingMinions.has(card.card_id)) Object.assign(entity, {
      attack: card.attack, health: card.health, maxHealth: card.max_health ?? card.health,
      golden: !!card.golden, taunt: !!card.taunt, divineShield: !!card.divine_shield,
      windfury: !!card.windfury, reborn: !!card.reborn, races: normalize(definition.races),
    });
    // Current stats already include permanent and temporary recruit buffs.
    // Keep their identities for handlers needing enchantment provenance.
    entity.enchantments = (card.enchantments ?? []).map((buff, index) => ({
      cardId: buff.blood_gem ? 'BG20_GEMe' : buff.source_id,
      entityId: entityId * 1000 + index + 1,
      tagScriptDataNum1: buff.attack, tagScriptDataNum2: buff.health,
      temporary: !!buff.temporary,
    })).filter(e => e.cardId);
    return entity;
  }

  input({ player, opponent, trials = 128 }) {
    this.validateSnapshot(player);
    this.validateSnapshot(opponent);
    if ([...player.validTribes].sort().join() !== [...opponent.validTribes].sort().join()) throw new Error('Opponents belong to different tribe lobbies');
    if (!Number.isInteger(trials) || trials < 1) throw new Error('Positive integer trials required');
    const side = (snapshot, base) => ({
      board: snapshot.board.map((c, i) => this.entity(c, base + i)),
      player: { cardId: snapshot.heroId, entityId: base - 1,
        hpLeft: snapshot.health ?? 60, tavernTier: snapshot.tavernTier,
        heroPowers: [], questEntities: [], questRewards: [], trinkets: [], secrets: [],
        enchantments: [], globalInfo: {},
        hand: snapshot.hand.map((c, i) => this.entity(c, base + 20 + i)) },
    });
    return { playerBoard: side(player, 100), opponentBoard: side(opponent, 200),
      options: { numberOfSimulations: trials, maxAcceptableDuration: 60000,
        intermediateResults: 0, includeOutcomeSamples: false, skipInfoLogs: true,
        applyDamageCap: false },
      gameState: { currentTurn: 1, validTribes: player.validTribes.map(t => Race[t]),
        anomalies: [], numberOfPlayersAlive: 8 } };
  }

  evaluate({ player, opponent, trials = 128, seed = 1 }) {
    const input = this.input({ player, opponent, trials });
    const result = withRng(createSeededRng(seed), () => {
      const iterator = simulateBattle(structuredClone(input), this.cards, this.context(player.validTribes));
      let next = iterator.next();
      while (!next.done) next = iterator.next();
      return next.value;
    });
    const n = result?.won + result?.tied + result?.lost;
    if (n !== trials) throw new Error(`Incomplete opening combat: ${n}/${trials}`);
    return { score: (result.won + 0.5 * result.tied) / n,
      simulations: { won: result.won, tied: result.tied, lost: result.lost, n, seed,
        meanNetDamage: (result.damageWon - result.damageLost) / n },
      scope: OPENING_SCOPE, actualHandIncluded: true,
      recruitReturnState: { nextTurnStateAvailable: false, combatChangesAppliedToRecruitment: false } };
  }
}
