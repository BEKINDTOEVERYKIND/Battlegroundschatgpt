# Training design

## Objective and implementation boundary

The target is a strong **Solo Battlegrounds player on the current patch**, with
transfer to future rotations. A trained positioning component is an initial
deliverable, not evidence of full-game playing strength.

The engine is the first bottleneck. A fast policy trained against wrong shop
rules, obsolete cards, missing spells, or incorrect seasonal effects learns
the wrong game. The release must report separately:

1. Source-data coverage: which active cards, modes, and balance changes were verified.
2. Engine coverage: which transitions and effects are implemented and tested.
3. Training scope: which decisions and state distributions the model actually saw.
4. Measured strength: held-out results, baseline, uncertainty, and sample count.

## Reuse across rotations

The first policy uses shared statistics, tribes, mechanics, and board positions,
not a table of card-ID-specific action scores. New cards with familiar mechanics
can therefore pass through the same encoder immediately. Checkpoints contain the
feature schema; incompatible warm starts are rejected. This makes transfer
possible; its benefit must still be measured on a later rotation.

Full-game learning should extend this to a shared entity encoder for the hand,
board, shop, hero, effects, and observed opponents. An autoregressive action
head selects action type, entity, target, and position under legal-action masks.
The encoder and value trunk persist; new action/mechanic adapters can be trained
without discarding the trunk. Text or structured effect features should augment
the representation when boolean mechanic flags cannot distinguish effects.

Keep a stable ruleset fingerprint in every episode, checkpoint, evaluation, and
replay. Old episodes may teach general representations but must not contribute
off-policy value targets as though their transition rules were current. After a
rotation, regenerate decisions with the new engine and compare warm-start vs
fresh initialization at equal simulation budgets and independent seeds.

## Positioning curriculum

Use the maintained Firestone combat simulator with frozen reference data.
Sample only explicitly eligible current cards. Generate complete board scenarios
before assigning train/validation/test, so alternate permutations of the same
board cannot leak across splits. Use different RNG streams for training labels
and final combat evaluation. Store the actual boards, labels, simulator version,
seed, and sampling restrictions alongside the model.

The supervised target is expected combat score (win + half a tie), not final
lobby placement. Compare learned ordering against the incoming order and simple
attack/taunt heuristics on the same held-out opponents, with confidence intervals
clustered by scenario. Independently re-simulate selected orders with a larger
budget; selecting the largest noisy label is not a trustworthy evaluation.

Synthetic snapshot results do not establish performance against human players.
As real replays become available, evaluate realistic buff sizes, synergies,
opponent uncertainty, hero effects, and seasonal effects separately.

## Full-game environment acceptance

A recruit engine must support eight players, the finite shared pool, lobby tribe
selection, hero selection and powers, shop offers, buying/selling/playing,
targeting, refresh/freeze, upgrading, triples/discover, hand and board limits,
spells, turn/start/end effects, combat persistence, pairing/ghosts, damage caps,
and placement/tiebreaks. Seasonal systems are explicit modules selected by the
verified manifest, never inferred from old game data.

For the current season this includes Dark Gifts, Trinkets, Activate, Lockbox, and
Fishbait where applicable. Any unavailable handler blocks full-game training;
silently resolving it as a vanilla minion or a no-op is forbidden.

Conformance fixtures must cover each implemented effect plus interactions and
state invariants. Differential combat checks against a second simulator help,
but agreement between simulators does not supersede observed client behavior.

Every rollout must use the shared [timed recruit wrapper](turn-budgets.md).
Finite player-specific action and time budgets apply to choices and movement as
well as economic actions, and cannot reset when a plan changes. The policy and
value model must observe remaining time/actions. End the turn at exhaustion with
validated pending-choice timeout semantics. Infinite-resource compositions still
have finite execution time. Default limits are one complete action per second
with a five-second reserve; additional command delays are explicit assumptions.
The currently audited recruitment scaffold has no validated timer integration.

## Full-game learning and promotion

Once the environment passes coverage, use masked PPO or another on-policy
actor-critic against a league of frozen historical and heuristic opponents.
Train for final placement utility; maintain a separately evaluated top-four or
first-place objective rather than silently changing utility. Persistent scaling
and economy require full episodes; one-combat labels cannot train those choices.

Opponent observations must match the information a player had at the decision.
Do not feed hidden hands, private shop offers, exact future pairings, or freshly
updated opponent boards into a player policy. The simulator may access hidden
state for transitions, not for action selection.

Promote only after a predeclared, paired held-out evaluation shows improvement
against multiple fixed opponents and the previous champion. Keep all candidates,
seeds, uncertainty intervals, failed attempts, and readable decision traces.
Do not report human MMR from simulator-only results.

## Next concrete milestone

Bring the best recruit-engine scaffold to the exact current ruleset; remove
silent fallback paths, implement uncovered active effects, and validate the
full-game contract. Only then launch and report full-game self-play training.
