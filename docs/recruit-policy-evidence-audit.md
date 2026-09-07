# Independent recruitment-policy evidence checks

`scripts/audit_recruit_policy_run.py --run runs/<completed-experiment>` audits the
new behavior-cloning and policy-gradient formats. It refuses to inspect an
unfinished run's validation or test evidence. It does not import the trainer,
execute archived source, rerun the simulator, or select another model.

The audit verifies every planned policy, episode seed and sample exactly once.
It recomputes complete trajectory scores from both sampled combat receipts,
then independently recomputes the reported paired means and 10,000 bootstrap
intervals. It checks recorded checkpoints and archived execution-source hashes,
independent seed families, accepted expert training counts, or whole-batch
gradient rejection and update counts. The output records the hashes of every
consumed evidence file.

Failed samples remain separate from actual combat outcomes. The experiment's
declared rule assigns zero to an entire episode if any requested sample fails;
the audit reports that failure-penalized episode mean alongside the actual mean
of completed samples. Own action counts, all-player and own timing violations,
and commands in failed partial trajectories are kept distinct. For accepted
expert training, it also reports the Tier-1 minions and spells actually visible
in player-zero board, hand and shop observations.

Two explicit restricted modes have separate output filenames:

- `--evaluation-only` writes `independent_evaluation_audit.json` and makes no
  claim that the training archive passed verification.
- `--verified-recovery` writes `independent_recovery_audit.json`. It accepts a
  separately recovered BC expert archive only after the recovery reproduced
  every original candidate checkpoint byte-for-byte. It preserves the fact
  that the original archive is incomplete.

`scripts/recover_recruit_expert_archive.py` regenerates only the original expert
training seeds into a new `recovered_expert_data/` directory. It compares every
recoverable original trajectory, refits all preregistered BC checkpoints using
the original recipe, and creates a reusable data manifest only when all original
checkpoint SHA256 values match. It leaves the original bytes unchanged and
does not read holdout trajectories.

These checks verify recorded evidence and deterministic reproduction. They do
not establish full-game rules correctness, currentness on a later date, or
competitive Battlegrounds strength.
