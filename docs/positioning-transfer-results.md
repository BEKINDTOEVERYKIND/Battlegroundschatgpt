# Corrected-pool positioning transfer results

The completed experiment found a positioning benchmark advantage for both
scratch and warm-start **hybrid policies**, each combining neural proposals with
the same finite simulation search. It did **not** establish an advantage for
warm starting over training from scratch. These results do not establish
full-game playing strength, recruiting quality, or a human MMR.

[Training run 33969375049](https://github.com/BEKINDTOEVERYKIND/Battlegroundschatgpt/actions/runs/33969375049)
completed successfully on September 5, 2026 at 14:22 UTC. Its recorded experiment
runtime was 2,715.13 seconds (45.25 minutes), including label generation, both
training arms, search, and fresh evaluation. The source commit was
`e8a05aa3ab467304fa8901b430a184937e43aab6`.

The [declared job](../runs/20260905-corrected-positioning-transfer-v1/job.json)
used Solo patch **36.4.2.251332**, Firestone combat **1.1.750**, and corrected
adapter **`firestone-combat-v2-tribes`**. It generated 8,000 synthetic board
scenarios: 6,400 training, 800 validation, and 800 final test. Reading the
[preserved dataset](../runs/20260905-corrected-positioning-transfer-v1/generation/positions.jsonl.gz)
confirms that all **138 minions eligible for this restricted positioning
benchmark** occurred across player and opponent boards. This is a subset of the
246 active minions; the [coverage manifest](../runs/20260905-corrected-positioning-transfer-v1/generation/positions.jsonl.coverage.json)
records exclusions for unsupported effects and missing persistent, hand, or
seasonal context. The correction restores Mech sampling and combat tribe
semantics; it does not make the benchmark a complete current game simulator.

Both arms used the same labels and scenario split, 40 new training epochs, a
32-unit hidden layer, and model seed 17. The warm arm resumed the preserved
[earlier positioning checkpoint](../runs/20260905-positioning-v2/selected_positioning.json).
The evaluated policy searched its top three neural proposals together with the
attack and taunt-last baselines, deduplicated to at most five candidate orders,
using 128 simulations per order. Final evaluation used 1,024 simulations per
distinct selected order with a separate RNG stream. Scores below are
`P(combat win) + 0.5 × P(combat tie)`, expressed as percentages.

| Policy | Mean combat score | Difference from attack | Paired 95% interval |
| --- | ---: | ---: | ---: |
| Attack baseline | 51.697% | — | — |
| Scratch hybrid | 53.026% | +1.329 percentage points | +1.005 to +1.675 points |
| Warm-start hybrid | 53.128% | +1.431 percentage points | +1.113 to +1.764 points |

Attack was the strongest of the four predeclared baselines on these 800 test
scenarios. Both hybrids also had positive lower confidence bounds against
random, health, and taunt-last ordering, satisfying the saved positioning
benchmark gate. The complete comparisons, including all baselines and raw
scenario details, are in the
[scratch evaluation](../runs/20260905-corrected-positioning-transfer-v1/scratch/evaluation.json)
and [warm evaluation](../runs/20260905-corrected-positioning-transfer-v1/warm/evaluation.json).
The intervals use 10,000 paired bootstrap resamples with the scenario as the
sampling unit. Any advantage belongs to the combined network-and-search policy;
this experiment does not isolate a contribution from the network alone.

The [paired warm-minus-scratch comparison](../runs/20260905-corrected-positioning-transfer-v1/comparison.json)
was **+0.102 percentage points**, with a **95% interval of −0.126 to +0.327
points**. That interval includes zero. Equal new epoch counts do not establish
faster adaptation, and the warm model also carries computation from its earlier
training. This was transfer into a corrected current-pool benchmark, not a test
on a subsequent rotation. Neither checkpoint was promoted based on these final
test results.

| Recorded simulation work | Combat trials |
| --- | ---: |
| Shared label generation | 12,288,000 |
| Scratch candidate search | 344,064 |
| Scratch fresh evaluation | 2,787,328 |
| Warm candidate search | 343,168 |
| Warm fresh evaluation | 2,769,920 |
| Total executed across these stages | 18,532,480 |

Generation comes directly from the
[generation manifest](../runs/20260905-corrected-positioning-transfer-v1/generation/positions.jsonl.meta.json).
Search counts sum `search_outcomes[*].simulations.n` in each arm's frozen
selections. Fresh counts sum `simulations.n` once per distinct `candidateIndex`
within each scenario, matching the evaluator's cache. Both arms actually ran
their own evaluation, so shared baseline simulations across arms are counted
twice in this execution total. The evidence is preserved in the
[scratch directory](../runs/20260905-corrected-positioning-transfer-v1/scratch/)
and [warm directory](../runs/20260905-corrected-positioning-transfer-v1/warm/),
including frozen choices, fresh outcomes, training reports, and compressed logs.

The [scratch checkpoint](../runs/20260905-corrected-positioning-transfer-v1/scratch/positioning.json)
and [warm checkpoint](../runs/20260905-corrected-positioning-transfer-v1/warm/positioning.json)
remain available for preregistered follow-up experiments. The
[artifact ingestion manifest](../runs/20260905-corrected-positioning-transfer-v1/artifact_ingest.json)
binds all 48 original files to their preserved representations and records the
verified original ZIP SHA256
`b30e74f5378355d18aae5975ab1f8fd2c6d80c84f43f7f5e22bd2707161908b6`.
The exact dataset restores from gzip; no successful training was rerun merely
to recover its results.
