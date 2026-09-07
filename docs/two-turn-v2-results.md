# Two-turn policy-state experiment, 6 September 2026

The [completed GitHub run](https://github.com/BEKINDTOEVERYKIND/Battlegroundschatgpt/actions/runs/34039468068)
finished at 15:20:17 UTC. Its execution succeeded, but its learned recruitment
policy performed substantially worse than the practical heuristic. It is not
promoted and must not become the behavior checkpoint for another iteration by
default.

The run collected decisions reached by the previous frozen policy, then labeled
each shortlisted action by finishing both turns with the practical heuristic.
It warm-started the previous 16-unit model, compared epochs 5 and 15 on validation
labels, and froze the selected checkpoint before 32 fresh paired test attempts.
Both test policies acted at every player-zero decision, with eight samples per
policy and seed.

| Measurement | Result |
| --- | ---: |
| Accepted training trajectories | 43 of 80 |
| Training / validation decisions | 918 / 202 |
| Complete paired test episodes | 31 of 32 |
| Model mean two-combat score | 0.1391129 |
| Practical heuristic mean score | 0.5614919 |
| Model minus practical | −0.4223790 |
| Paired 95% bootstrap interval | [−0.5241935, −0.3135081] |
| Model Freeze commands | 7,632 |
| Model Buy / Play commands | 360 / 160 |
| Recorded action-budget violations | 0 |

The score is the mean of the first and second sampled combat outcomes, with
win = 1, tie = 0.5, and loss = 0. It is neither lobby placement nor an MMR
estimate. Confidence intervals resample complete paired episodes after averaging
their eight samples.

Thirty-two training attempts ended at an unsupported pending-choice timeout,
one reached the unverified off-board Scarlet effect, and four reached the
unverified Tavern Fugitive spell trigger. One test attempt was rejected because
the practical policy reached a pending-choice timeout. These rejected attempts
are reported separately; the comparison is conditional on complete supported
episodes.

The zero timing violations do not make the behavior useful. The model spends
much of its finite clock repeatedly issuing Freeze. The experiment demonstrates
that collecting more states from the previous policy, while retaining the
practical continuation labels, did not repair full-policy behavior.

The game scope remains the current complete Tier-1 pool, two recruit turns,
eight Patchwerk players and declared synthetic timers. The action profile is
one complete command per second, additional command costs, and a five-second
reserve. Tier-2 shop transitions, ordinary triples and unverified contextual
effects remain explicit frontiers. The current snapshot passed live checks
before and after the original run; preservation does not re-label it as a later
snapshot.

The original artifact contains 16 files and 61,091,608 bytes. Its upload SHA256
is `00b14a654be739e3c66aa872bef9f8df1ebb664111e89d4042e70515f86c1fd7`.
All original files are preserved in
[`runs/20260906-two-turn-policy-states-v2/`](../runs/20260906-two-turn-policy-states-v2/).
The [independent ingest run](https://github.com/BEKINDTOEVERYKIND/Battlegroundschatgpt/actions/runs/34088968487)
passed on 7 September and published commit
`dec16ffd73c8d5e4d9bf1fee85ca40f6555c9390`. It checked the exact ZIP,
source commit, runner configuration and behavior checkpoint, recomputed every
evaluation episode score from both sampled combat receipts, and checked raw
counts and split identities. Its manifest accounts for every original byte,
including the retained checkpoints and compressed traces. The score means,
paired mean difference, action counts and timing count above therefore agree
with independently checked raw evidence. The bootstrap interval is preserved
from the original evaluation report; ingestion does not rerun that bootstrap.
