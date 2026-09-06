# Bounded GitHub training

The **Bounded current-pool training** workflow runs one finite experiment when
`config/training-job.json` changes on `main`, or when manually started from Actions.
Ordinary code commits do not retrigger it. There is no recurring schedule.

The declared experiment generates 8,000 new current-pool positioning scenarios
with corrected Mech handling, 12 candidate orders, and 128 label combats per
candidate. It trains one model from scratch and one from the hash-pinned existing
checkpoint on exactly the same split data. Both use 40 epochs and the same finite
search policy, with 1,024 fresh combat trials per evaluated order. The 800 test
scenarios are not used for normalization, gradients, or checkpoint selection.
All settings are fixed before the run. The old checkpoint's original test boards
are not reused as the new dataset.

This tests whether the existing representation remains useful when adding the
Mechs missed by the earlier sampler and correcting lobby semantics. It is not
evidence about a future seasonal rotation, human MMR, or whole-game performance.
Equal training epoch counts alone cannot establish faster adaptation.

Both arms retain checkpoints, optimizer/RNG state, source hashes, frozen choices,
fresh results, paired confidence intervals, and live-source checks. Any changed
or unavailable watched live source stops current training. A failed stage retains
its logs and completed artifacts. Neither model is automatically promoted.

The training step has a 320-minute limit and the job a 350-minute limit. Workflow
permissions are read-only. Artifacts are retained in GitHub Actions for 90 days;
completed results intended for long-term use should be reviewed and committed
to the repository before that retention expires. The workflow itself does not
purchase compute, change account settings, send messages, or write commits.

Run [33969375049](https://github.com/BEKINDTOEVERYKIND/Battlegroundschatgpt/actions/runs/33969375049)
completed successfully on 5 September 2026 at 14:22:02 UTC, after about 46 minutes.
Its 48-file artifact is `training-33969375049-1` (ID `9971076392`), with ZIP SHA256
`b30e74f5378355d18aae5975ab1f8fd2c6d80c84f43f7f5e22bd2707161908b6`.
Completion alone is not a positive strength or transfer result.

A separate, explicitly configured **Preserve completed training results** job
can verify and commit that finished artifact. It checks the source run, commit,
artifact identity, ZIP digest, and paths before preserving the evidence. This job
has read access to Actions and write access to repository contents; its push is
an ordinary fast-forward and never reruns the training. Its exact source and
destination are declared in `config/result-ingest-job.json`.
Future training runs also print their compact comparison to the job log and
GitHub step summary, so inspecting scores does not depend on downloading a ZIP.

To inspect the exact commands locally:

```bash
python scripts/run_training_job.py --out /absolute/new/run --dry-run
```

Remove `--dry-run` to execute. The output must be a new directory. The runner
validates the checkpoint hash and bounded settings before doing work. See
[GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
and [artifact retention](https://github.com/actions/upload-artifact#retention-period).
