"""Synthetic report-integrity fixtures; not Battlegrounds performance evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_fresh_evaluation import summarize


class SearchReportTests(unittest.TestCase):
    def test_names_combined_policy_and_rejects_search_rng_reuse(self):
        policy = 'neural_proposals_plus_simulation_search'
        names = ['model', 'random', 'attack', 'health', 'taunt_last']
        frozen = {'scenario_id': 'synthetic-fixture', 'split': 'test',
                  'selections': {name: 0 for name in names},
                  'provenance': {'evaluated_policy': policy, 'label_combat_seed': 1,
                                 'search_seed': 3, 'search_trials': 128, 'search_top_k': 3,
                                 'proposal_file_sha256': 'fixture-proposal',
                                 'cards_sha256': 'fixture-cards', 'ruleset_sha256': 'fixture-rules',
                                 'checkpoint_sha256': 'fixture-model', 'dataset_sha256': 'fixture-data',
                                 'primary_baseline': 'attack', 'all_baselines': names[1:]},
                  'search_outcomes': [{'candidateIndex': i, 'score': 0.5,
                                       'simulations': {'n': 128, 'seed': 3}} for i in range(3)]}
        fresh = {'scenario_id': 'synthetic-fixture', 'split': 'test',
                 'scores': {name: {'candidateIndex': 0, 'score': 0.5,
                                   'simulations': {'won': 10, 'tied': 0, 'lost': 10, 'n': 20, 'seed': 2}}
                            for name in names},
                 'metadata': {'trials': 20, 'combatSeed': 2, 'engine': 'fixture',
                              'cardsSha256': 'fixture-cards', 'rulesetSha256': 'fixture-rules',
                              'evaluatedPolicy': policy}}
        with tempfile.TemporaryDirectory() as directory:
            selections = Path(directory) / 'frozen.jsonl.gz'
            data = (json.dumps(frozen) + '\n').encode()
            selections.write_bytes(gzip.compress(data, mtime=0))
            fresh['metadata']['selectionsSha256'] = hashlib.sha256(data).hexdigest()
            output = Path(directory) / 'fresh.jsonl'
            output.write_text(json.dumps(fresh) + '\n')
            report = summarize(output, selections, bootstrap_samples=100)
            self.assertEqual(report['evaluated_policy'], policy)
            self.assertIn('cannot be attributed to the neural model alone', report['policy_interpretation'])
            self.assertFalse(report['results']['test']['positioning_benchmark_gate']['passes'])
            fresh['metadata']['combatSeed'] = 3
            output.write_text(json.dumps(fresh) + '\n')
            with self.assertRaisesRegex(ValueError, 'search RNG'):
                summarize(output, selections, bootstrap_samples=100)


if __name__ == '__main__':
    unittest.main()
