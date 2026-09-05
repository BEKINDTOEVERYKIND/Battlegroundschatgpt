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

To inspect the exact commands locally:

```bash
python scripts/run_training_job.py --out /absolute/new/run --dry-run
```

Remove `--dry-run` to execute. The output must be a new directory. The runner
validates the checkpoint hash and bounded settings before doing work. See
[GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
and [artifact retention](https://github.com/actions/upload-artifact#retention-period).
