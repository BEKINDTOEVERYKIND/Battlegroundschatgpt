/** Opt-in Scarlet one-time marker transport; old workers are unchanged. */
import fs from 'node:fs';
import readline from 'node:readline';
import { pathToFileURL } from 'node:url';
import { OpeningTransitionFirestone } from './opening-transition-firestone.mjs';

export const EFFECT_TRANSPORT_VERSION = 'current-hand-scarlet-opening-v1';

export class HandScarletFirestone extends OpeningTransitionFirestone {
  static fromFiles(cardsPath = 'data/reference_cards.json', rulesetPath = 'data/ruleset.json') {
    const checked = OpeningTransitionFirestone.fromFiles(cardsPath, rulesetPath);
    return new HandScarletFirestone({ referenceCards: JSON.parse(fs.readFileSync(cardsPath, 'utf8')),
      ruleset: checked.ruleset, cardsHash: checked.cardsHash, rulesetHash: checked.rulesetHash });
  }

  validateSnapshot(snapshot) {
    super.validateSnapshot(snapshot);
    if (snapshot.effect_transport_version !== EFFECT_TRANSPORT_VERSION) throw new Error('Scarlet effect version required');
    for (const card of [...snapshot.board, ...snapshot.hand]) {
      if (card.card_id === 'BG35_814' && typeof card.scarlet_trigger_consumed !== 'boolean') {
        throw new Error('Explicit Scarlet one-time marker required');
      }
    }
  }

  entity(card, entityId) {
    const result = super.entity(card, entityId);
    if (card.card_id === 'BG35_814') {
      if (typeof card.scarlet_trigger_consumed !== 'boolean') throw new Error('Explicit Scarlet one-time marker required');
      // abiity is the pinned Firestone API's spelling. Both channels are
      // transported because its handler consults charges AND enchantments.
      result.abiityChargesLeft = card.scarlet_trigger_consumed ? 0 : 1;
      if (card.scarlet_trigger_consumed && !result.enchantments.some(e => e.cardId === 'BG35_814e')) {
        result.enchantments.push({ cardId: 'BG35_814e', originEntityId: entityId,
          timing: 0, tagScriptDataNum1: -1, tagScriptDataNum2: -1 });
      }
    }
    return result;
  }

  sample(request) {
    if (request.effect_transport_version !== EFFECT_TRANSPORT_VERSION) throw new Error('Scarlet request version required');
    return { ...super.sample(request), effect_transport_version: EFFECT_TRANSPORT_VERSION };
  }

  evaluate(input) {
    return { ...super.evaluate(input), scope: EFFECT_TRANSPORT_VERSION,
      effect_transport_version: EFFECT_TRANSPORT_VERSION };
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const engine = HandScarletFirestone.fromFiles();
  for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    try { process.stdout.write(`${JSON.stringify(engine.sample(JSON.parse(line)))}\n`); }
    catch (error) { process.stdout.write(`${JSON.stringify({ error: String(error.message ?? error) })}\n`); }
  }
}
