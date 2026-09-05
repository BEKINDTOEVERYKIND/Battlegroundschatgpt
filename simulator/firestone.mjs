/** Offline, frozen-data adapter for Firestone's maintained combat simulator. */
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { AllCardsService, Race } = require('@firestone-hs/reference-data');
const { CardsData } = require('@firestone-hs/simulate-bgs-battle/dist/cards/cards-data.js');
const { cardMappings } = require('@firestone-hs/simulate-bgs-battle/dist/cards/impl/_card-mappings.js');
const { simulateBattle, createSeededRng, withRng } = require('@firestone-hs/simulate-bgs-battle');
export const ENGINE_VERSION = require('@firestone-hs/simulate-bgs-battle/package.json').version;
export const REFERENCE_VERSION = require('@firestone-hs/reference-data/package.json').version;
export const TRIBES = ['BEAST', 'DEMON', 'DRAGON', 'ELEMENTAL', 'MECHANICAL', 'MURLOC', 'NAGA', 'PIRATE', 'QUILBOAR', 'UNDEAD'];

export function sha256(bytes) { return crypto.createHash('sha256').update(bytes).digest('hex'); }
export function readJson(file) { return JSON.parse(fs.readFileSync(file, 'utf8')); }
export function seeded(seed) { return createSeededRng(seed); }
export function shuffle(items, rng) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; --i) {
    const j = Math.floor(rng.next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export function makeEntity(card, entityId, bonus = 0) {
  const mechanics = new Set(card.mechanics ?? []);
  const text = (card.text ?? '').replace(/\{(\d+)\}/g, (token, index) => card[`scriptDataNum${Number(index) + 1}`] ?? token);
  const scriptData = Object.fromEntries([1, 2, 3, 4, 5, 6].filter(i => card[`scriptDataNum${i}`] != null)
    .map(i => [`scriptDataNum${i}`, card[`scriptDataNum${i}`]]));
  return {
    cardId: card.id, entityId, name: card.name, text, ...scriptData, attack: (card.attack ?? 0) + bonus,
    health: (card.health ?? 1) + bonus, tavernTier: card.techLevel,
    races: card.races ?? [], golden: !!card.premium,
    taunt: mechanics.has('TAUNT'), divineShield: mechanics.has('DIVINE_SHIELD'),
    venomous: mechanics.has('VENOMOUS'), poisonous: mechanics.has('POISONOUS'),
    reborn: mechanics.has('REBORN'), windfury: mechanics.has('WINDFURY'),
    cleave: mechanics.has('CLEAVE') || /also damages.*(?:next to|adjacent)/i.test(card.text ?? ''), deathrattle: mechanics.has('DEATHRATTLE'),
    avenge: mechanics.has('AVENGE'), frenzy: mechanics.has('FRENZY'),
    enchantments: [],
  };
}

export class FirestoneCombat {
  constructor({ referenceCards, ruleset, cardsHash = null, rulesetHash = null }) {
    if (!Array.isArray(referenceCards) || !referenceCards.length) throw new Error('Empty reference snapshot');
    this.ruleset = ruleset;
    this.cardsHash = cardsHash;
    this.rulesetHash = rulesetHash;
    this.minionIds = new Set(ruleset.active?.minion_ids ?? []);
    this.shopMinionIds = new Set(ruleset.active?.shop_minion_ids ?? ruleset.active?.minion_ids ?? []);
    this.heroIds = new Set(ruleset.active?.hero_ids ?? []);
    if (!this.minionIds.size || !this.heroIds.size) throw new Error('Active Solo minion and hero allowlists are required');
    this.reference = new Map(referenceCards.map(c => [c.id, c]));
    this.dbf = new Map(referenceCards.map(c => [c.dbfId, c]));
    for (const id of [...this.minionIds, ...this.heroIds]) {
      if (!this.reference.has(id)) throw new Error(`Active card has no metadata: ${id}`);
    }
    // Keep tokens, enchantments and golden definitions available for scripted effects,
    // while overriding every random recruit pool from the official active allowlist.
    const frozen = referenceCards.map(c => ({ ...c, isBaconPool: this.shopMinionIds.has(c.id) && (c.techLevel ?? 0) <= 6 }));
    this.cards = new AllCardsService();
    this.cards.initializeCardsDbFromCards(frozen);
    this.contextCache = new Map();
  }

  static fromFiles(cardsPath = 'data/reference_cards.json', rulesetPath = 'data/ruleset.json') {
    const cardsBytes = fs.readFileSync(cardsPath);
    const ruleBytes = fs.readFileSync(rulesetPath);
    const ruleset = JSON.parse(ruleBytes);
    if (ruleset.status !== 'verified' && ruleset.combat_pool_verified !== true) throw new Error('Combat pool is not verified');
    const relative = path.relative(path.dirname(path.resolve(rulesetPath)), path.resolve(cardsPath)).replaceAll(path.sep, '/');
    const expectedHash = ruleset.checksums?.[relative];
    if (!expectedHash || expectedHash !== sha256(cardsBytes)) throw new Error(`Unverified or changed card snapshot: ${relative}`);
    return new FirestoneCombat({ referenceCards: JSON.parse(cardsBytes), ruleset,
      cardsHash: sha256(cardsBytes), rulesetHash: sha256(ruleBytes) });
  }

  isActiveMinion(cardId) {
    const card = this.reference.get(cardId);
    return this.minionIds.has(cardId) || !!(card?.premium && this.minionIds.has(this.dbf.get(card.battlegroundsNormalDbfId)?.id));
  }

  positioningEligibility(cardId) {
    const card = this.reference.get(cardId);
    if (!this.shopMinionIds.has(cardId) || !(card?.health > 0) || !(card.techLevel <= 6)) return 'outside normal Solo shop pool';
    const text = (card.text ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
    if (/your hand|in hand|this game|last combat|last turn|Blood Gem|trinket|Dark Gift|remember|Spellcraft/i.test(text)) return 'requires excluded hand, history, or seasonal context';
    if (/once this reaches|when this reaches|always has/i.test(text)) return 'requires initial threshold or stat-history reconstruction';
    const implementation = cardMappings[cardId];
    const behaviorSource = Object.values(implementation ?? {}).filter(v => typeof v === 'function').map(String).join('\n');
    if (/globalInfo|\.hand\b|scriptDataNum|remembered|\.memory\b/i.test(behaviorSource)) return 'handler requires excluded persistent context';
    if (/getRandomMinion|ghastcoilerSpawns|demonSpawns|pirateSpawns|beastSpawns/i.test(behaviorSource)) return 'random summon support closure has not been audited';
    const combatText = /Deathrattle|Avenge|Rally|Frenzy|Start of Combat|Whenever|attacks|takes damage|your other|adjacent/i.test(text);
    if (combatText && !implementation) return 'combat text lacks a registered per-card handler';
    return null;
  }

  context(validTribes) {
    const key = [...validTribes].sort().join(',');
    if (!this.contextCache.has(key)) {
      const data = new CardsData(this.cards, false);
      data.inititialize(validTribes.map(t => Race[t]), []);
      // Firestone's generic random tavern-spell method currently returns null;
      // never silently treat this as a correctly simulated effect.
      data.getRandomTavernSpell = () => { throw new Error('Unsupported Firestone random Tavern spell effect'); };
      data.getRandomMinorTimewarpCard = () => { throw new Error('Inactive Timewarped mechanic requested'); };
      const activeSpells = new Set(this.ruleset.active?.tavern_spell_ids ?? []);
      const getTwoCostSpells = data.getTwoCostBattlegroundsSpellIdsImplemented.bind(data);
      data.getTwoCostBattlegroundsSpellIdsImplemented = () => getTwoCostSpells().filter(id => activeSpells.has(id));
      this.contextCache.set(key, data);
    }
    return this.contextCache.get(key);
  }

  validateBoard(board, validTribes) {
    if (!Array.isArray(board) || board.length > 7) throw new Error('Combat board must have at most seven minions');
    for (const entity of board) {
      if (!this.isActiveMinion(entity.cardId)) throw new Error(`Inactive minion rejected: ${entity.cardId}`);
      if (!Number.isFinite(entity.attack) || entity.attack < 0 || !Number.isFinite(entity.health) || entity.health <= 0) {
        throw new Error(`Invalid minion stats: ${entity.cardId}`);
      }
      const races = this.reference.get(entity.cardId).races ?? [];
      if (races.length && !races.includes('ALL') && !races.some(r => validTribes.includes(r))) {
        throw new Error(`Minion tribe absent from lobby: ${entity.cardId}`);
      }
    }
  }

  evaluate({ board, opponent, validTribes, heroId, opponentHeroId = heroId, turn = 8,
    tavernTier = 4, opponentTavernTier = tavernTier, trials = 128, seed = 1,
    playerGlobalInfo = {}, opponentGlobalInfo = {} }) {
    if (!Number.isInteger(trials) || trials < 1) throw new Error('trials must be a positive integer');
    if (!Array.isArray(validTribes) || validTribes.length !== 5 || new Set(validTribes).size !== 5 || validTribes.some(t => !TRIBES.includes(t))) {
      throw new Error('A Solo lobby requires five distinct valid tribes');
    }
    if (!this.heroIds.has(heroId) || !this.heroIds.has(opponentHeroId)) throw new Error('Inactive hero rejected');
    if (![heroId, opponentHeroId].every(id => /Patchwerk/i.test(this.reference.get(id)?.name ?? ''))) {
      throw new Error('This adapter currently validates only Patchwerk positioning contexts');
    }
    this.validateBoard(board, validTribes);
    this.validateBoard(opponent, validTribes);
    const ids = [...board, ...opponent].map(c => c.entityId);
    if (new Set(ids).size !== ids.length || ids.some(id => !Number.isInteger(id))) throw new Error('Unique integer entity IDs required');
    const player = (id, tier, entityId, globalInfo) => ({
      cardId: id, entityId, hpLeft: 30, tavernTier: tier, heroPowers: [],
      questEntities: [], questRewards: [], trinkets: [], hand: [], secrets: [], enchantments: [], globalInfo,
    });
    // The positioning task is explicitly conditioned on no combat hero-power,
    // trinket, gift, or quest effects; those systems are not approximated here.
    const input = {
      playerBoard: { board, player: player(heroId, tavernTier, 1, playerGlobalInfo) },
      opponentBoard: { board: opponent, player: player(opponentHeroId, opponentTavernTier, 2, opponentGlobalInfo) },
      options: { numberOfSimulations: trials, maxAcceptableDuration: 60000, intermediateResults: 0,
        includeOutcomeSamples: false, skipInfoLogs: true, applyDamageCap: false },
      gameState: { currentTurn: turn, validTribes: validTribes.map(t => Race[t]), anomalies: [], numberOfPlayersAlive: 8 },
    };
    const result = withRng(createSeededRng(seed), () => {
      const iterator = simulateBattle(structuredClone(input), this.cards, this.context(validTribes));
      let next = iterator.next();
      while (!next.done) next = iterator.next();
      return next.value;
    });
    const completed = result?.won + result?.tied + result?.lost;
    if (completed !== trials) throw new Error(`Incomplete simulation: ${completed}/${trials}`);
    const meanNetDamage = (result.damageWon - result.damageLost) / completed;
    if (!Number.isFinite(meanNetDamage)) throw new Error('Nonfinite combat damage output');
    return {
      score: (result.won + 0.5 * result.tied) / completed,
      simulations: { won: result.won, tied: result.tied, lost: result.lost, n: completed,
        meanNetDamage, seed },
    };
  }

  supportReport() {
    const rows = [...this.minionIds].map(id => {
      const card = this.reference.get(id);
      return { id, name: card.name, techLevel: card.techLevel, races: card.races ?? [],
        mappedHandler: !!cardMappings[id], mechanics: card.mechanics ?? [],
        positioningExclusion: this.positioningEligibility(id) };
    });
    return { engine: ENGINE_VERSION, referencePackage: REFERENCE_VERSION, cardsSha256: this.cardsHash,
      rulesetSha256: this.rulesetHash, activeMinions: rows.length,
      note: 'A mapped handler is evidence of implementation, not a guarantee of correct behavior. Recruit-only effects may need no combat handler.', rows };
  }
}
