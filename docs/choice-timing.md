# Explicit choice completion within a finite turn

`bg_ai.choice_timing.ChoiceTimedTwoTurnOpeningEngine` is a separately versioned
wrapper for the current two-turn Tier 1 fixture. Its version is
`visible-opening-choice-reservation-v1`. Previously saved experiments continue
to use their original timing wrapper and retain their original interpretation.

The prior wrapper checked whether the next command fitted. A cast could fit
while its required target selection did not, reaching an unverified client
timeout and rejecting the entire trajectory. This wrapper checks the complete
known command sequence before allowing its first command. It does not simulate
what the client does when a choice expires.

| Visible action | Commands after the initiator | Default complete cost |
| --- | --- | --- |
| Alliance Flag cast | Choose One, then choose a target | 5,500 ms; 3 commands |
| Fortify, Tavern Dish Banana, Blood Gem, Mini-Trident cast | Choose a target | 3,500 ms; 2 commands |
| A New Sprout cast | At most one Discover selection | At most 3,500 ms; 2 commands |
| Suspicious Prisonguard Activate | Choose another friendly board minion | 3,500 ms; 2 commands |
| Current Tier 1 minion play, including all three Battlecries | None | 1,500 ms; 1 command |

Costs use the existing one-command-per-second profile and its extra delays.
They are explicit training assumptions, not measurements of client animations.
The five-second safety reserve is deducted before checking any sequence.
Both total milliseconds and the per-turn command limit must fit. Reservations
do not precharge or refund time: every cast, activation and choice still passes
through the shared `TurnBudget.charge` exactly once. A fork retains the same
spent clock and remaining choice sequence.

A New Sprout reserves the possible Discover selection from the visible spell
identity. The mask never examines the shared pool, RNG or opponent zones. If
the actual engine produces no Discover options, it releases the reservation
without inventing a selection or charging a nonexistent command. Resolving a
Discover can still encounter the existing unsupported-triple frontier.

All targets come unchanged from the underlying adapter's legal list. The
wrapper does not remove unsupported card interactions, triple acquisitions,
Freeze, or temporary Magnetic transfers to improve acceptance rates. It only
removes a command if its audited completion cannot fit the remaining budget.
Generated Gems, Spellcraft and acquired minions remain in hand until later
explicit commands play them; those optional commands are not part of the
Battlecry reservation.

If a new observation shortens the timer below the remaining reserved sequence,
the rollout still aborts. It neither chooses automatically nor completes the
sequence for free. A failed rollout cannot be forked and retried with a fresh
budget. Full-game readiness remains false.

The command graph was audited against the existing opening adapter and pinned
HSBRSIM revision `4e0525a198352557ef816bd3a53566d6ffd4c164`, using current
reference SHA-256
`88dd032aa56ddb8a176ace6d5bafa48b2d78b4049fd2b97c0e6a00e663f1ad14`.
A changed reference snapshot requires a new audit before this wrapper is used.
No historical cards are introduced by the timing table.

`tests/test_choice_timing.py` runs the real current-card engine at the exact
completion boundary and one millisecond below it. It checks all 22 current
Tier 1 minion plays, all required target and branch stages, original legal
targets, separate charges, independent forks, empty-Discover handling, timer
shrinkage, continued finite Freeze actions, and unchanged unsupported-triple
failures. Existing first-version timing behavior has a regression check.
