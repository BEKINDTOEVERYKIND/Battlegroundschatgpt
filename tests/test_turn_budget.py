"""Finite recruit-clock contracts using an explicitly synthetic toy adapter.

The fake report validates only this test adapter, never the current card pool.
"""

import unittest

from bg_ai.recruiting import (
    RecruitAction, RecruitCoverageError, RecruitObservation, ruleset_digest,
)
from bg_ai.timed_recruiting import TimedRecruitEngine
from bg_ai.turn_budget import TimingProfile, TurnBudget


TOY_RULESET = {"name": "synthetic-clock-test", "contains_actual_cards": False}
BUY = RecruitAction("buy", entity_id="toy-shop-entity")
SELL = RecruitAction("sell", entity_id="toy-board-entity")
CHOOSE = RecruitAction("choose", choice_index=0)
END = RecruitAction("end_turn")


def observation(*, player=0, turn=1, remaining=3500, actions=(BUY, END)):
    return RecruitObservation(
        player_id=player, turn=turn, public_state={}, private_state={},
        legal_actions=actions, ruleset_sha256=ruleset_digest(TOY_RULESET),
        time_remaining_ms=remaining,
    )


class ToyRecruitEngine:
    """Script transitions; timeout resolution belongs to the adapter itself."""

    def __init__(self, initial, *, after_steps=(), after_expirations=()):
        self.initial = initial
        self.after_steps = list(after_steps)
        self.after_expirations = list(after_expirations)
        self.steps = []
        self.expirations = 0
        self.report = {
            "ruleset_sha256": ruleset_digest(TOY_RULESET),
            "full_game_ready": True,
            "conformance_validated": True,
            "blockers": [],
            "turn_timing": {
                "actual_player_window_validated": True,
                "pending_choice_timeout_validated": True,
            },
        }

    def coverage(self):
        return self.report

    def reset(self, *, seed):
        self.steps.clear()
        self.expirations = 0
        return self.initial

    def step(self, action):
        self.steps.append(action)
        index = len(self.steps) - 1
        return self.after_steps[index] if index < len(self.after_steps) else None

    def expire_turn(self):
        index = self.expirations
        self.expirations += 1
        return (self.after_expirations[index]
                if index < len(self.after_expirations) else None)

    def final_placements(self):
        return {0: 1.0}


class RepeatingBuySellEngine(ToyRecruitEngine):
    """Unlimited money and a nondecreasing adapter timer must still terminate."""

    def __init__(self, remaining):
        self.remaining = remaining
        super().__init__(observation(remaining=remaining, actions=(BUY,)))

    def step(self, action):
        self.steps.append(action)
        return observation(remaining=self.remaining,
                           actions=(SELL if action == BUY else BUY,))


class TurnBudgetTests(unittest.TestCase):
    def profile(self, *, rate=2, reserve=0, extra=None):
        return TimingProfile(rate, reserve,
                             {"buy": 0, "sell": 0, "choose": 2000}
                             if extra is None else extra)

    def test_exact_boundary_includes_last_affordable_action(self):
        budget = TurnBudget(self.profile(reserve=250), 1750)
        self.assertEqual(budget.action_limit, 3)
        self.assertEqual([budget.charge("buy") for _ in range(3)], [500] * 3)
        self.assertEqual(budget.remaining_ms, 0)
        self.assertEqual(budget.remaining_actions, 0)
        self.assertFalse(budget.can_afford("sell"))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            budget.charge("sell")
        self.assertEqual(budget.actions_used, 3)

    def test_animation_delay_can_exhaust_time_before_global_action_count(self):
        budget = TurnBudget(self.profile(), 3000)
        self.assertEqual(budget.action_limit, 6)
        self.assertEqual(budget.charge("choose"), 2500)
        self.assertFalse(budget.can_afford("choose"))
        self.assertTrue(budget.can_afford("buy"))
        budget.charge("buy")
        self.assertEqual(budget.actions_used, 2)
        self.assertEqual(budget.remaining_actions, 0)

    def test_zero_budget_and_reserve_larger_than_timer_do_not_allow_actions(self):
        for remaining in (0, 1, 499, 500):
            with self.subTest(remaining=remaining):
                budget = TurnBudget(self.profile(reserve=500), remaining)
                self.assertEqual(budget.remaining_ms, 0)
                self.assertEqual(budget.action_limit, 0)
                self.assertFalse(budget.can_afford("buy"))

    def test_insufficient_fraction_of_one_action_is_not_spendable(self):
        budget = TurnBudget(self.profile(reserve=500), 999)
        self.assertEqual(budget.remaining_ms, 499)
        self.assertEqual(budget.remaining_actions, 0)
        self.assertFalse(budget.can_afford("buy"))

    def test_observed_timer_can_tighten_but_cannot_refill(self):
        budget = TurnBudget(self.profile(reserve=500), 5500)
        budget.charge("buy")
        budget.observe_remaining(999999)
        self.assertEqual(budget.remaining_ms, 4500)
        budget.observe_remaining(1300)
        self.assertEqual(budget.remaining_ms, 800)
        budget.observe_remaining(5500)
        self.assertEqual(budget.remaining_ms, 800)
        budget.charge("buy")
        self.assertEqual(budget.remaining_ms, 300)
        self.assertEqual(budget.action_limit, 10)
        self.assertEqual(budget.remaining_actions, 0)

    def test_unknown_action_is_rejected_even_with_no_time_remaining(self):
        for available in (0, 3000):
            with self.subTest(available=available):
                budget = TurnBudget(self.profile(), available)
                with self.assertRaisesRegex(ValueError, "Unmapped action kind"):
                    budget.can_afford("new-seasonal-command")

    def test_search_fork_preserves_spent_and_observed_budget_without_shared_mutation(self):
        budget = TurnBudget(self.profile(reserve=500), 5500)
        budget.charge("buy")
        budget.charge("sell")
        budget.observe_remaining(2300)
        before = budget.snapshot()
        branch = budget.fork()
        self.assertEqual(branch.snapshot(), before)
        self.assertEqual(branch.remaining_ms, 1800)
        self.assertEqual(branch.actions_used, 2)
        branch.observe_remaining(999999)
        self.assertEqual(branch.remaining_ms, 1800)
        branch.charge("buy")
        self.assertEqual(branch.actions_used, 3)
        self.assertEqual(branch.remaining_ms, 1300)
        self.assertEqual(budget.snapshot(), before)
        branch_before = branch.snapshot()
        budget.observe_remaining(900)
        self.assertEqual(budget.remaining_ms, 400)
        self.assertEqual(branch.snapshot(), branch_before)

    def test_invalid_rates_and_millisecond_inputs_fail_closed(self):
        for rate in (0, -1, float("nan"), float("inf"), -float("inf"),
                     True, "2", 1001):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    self.profile(rate=rate)
        for value in (-1, float("nan"), float("inf"), 1.5, True, "1000", None):
            with self.subTest(milliseconds=value):
                with self.assertRaises(ValueError):
                    self.profile(reserve=value)
                with self.assertRaises(ValueError):
                    self.profile(extra={"buy": value})
                with self.assertRaises(ValueError):
                    TurnBudget(self.profile(), value)
                with self.assertRaises(ValueError):
                    TurnBudget(self.profile(), 1000).observe_remaining(value)

    def test_profile_cannot_be_mutated_through_source_mapping(self):
        source = {"buy": 0}
        profile = self.profile(extra=source)
        fingerprint = profile.fingerprint
        source["buy"] = 50000
        self.assertEqual(profile.action_ms("buy"), 500)
        self.assertEqual(profile.fingerprint, fingerprint)
        with self.assertRaises(TypeError):
            profile.extra_ms["buy"] = 50000


class TimedRecruitEngineTests(unittest.TestCase):
    def profile(self, *, reserve=500):
        return TimingProfile(1, reserve, {"buy": 0, "sell": 0, "choose": 2000})

    def wrapper(self, engine, *, profile=None):
        return TimedRecruitEngine(engine, profile or self.profile(), TOY_RULESET)

    def test_repeating_buy_sell_loop_always_ends_at_actual_turn_budget(self):
        for timer, expected in ((0, 0), (499, 0), (1500, 1), (5500, 5), (12500, 12)):
            with self.subTest(timer=timer):
                engine = RepeatingBuySellEngine(timer)
                timed = self.wrapper(engine)
                visible = timed.reset(seed=17)
                for _ in range(expected + 1):
                    if visible is None:
                        break
                    visible = timed.step(visible.legal_actions[0])
                self.assertIsNone(visible, "A repeating zero-gold cycle did not terminate")
                self.assertEqual(len(engine.steps), expected)
                self.assertEqual(engine.expirations, 1)
                self.assertEqual(engine.steps, [BUY if i % 2 == 0 else SELL
                                                for i in range(expected)])

    def test_interleaved_players_do_not_receive_new_budgets(self):
        engine = ToyRecruitEngine(
            observation(player=0),
            after_steps=[observation(player=1),
                         observation(player=0, remaining=999999),
                         observation(player=1, remaining=999999),
                         observation(player=0, remaining=999999),
                         observation(player=1, remaining=999999),
                         observation(player=0, remaining=999999)],
            after_expirations=[observation(player=1, remaining=999999), None],
        )
        timed = self.wrapper(engine)
        visible = timed.reset(seed=17)
        for player, remaining_actions in ((0, 3), (1, 3), (0, 2), (1, 2), (0, 1), (1, 1)):
            self.assertEqual(visible.player_id, player)
            self.assertEqual(visible.action_budget["remaining_actions"], remaining_actions)
            visible = timed.step(BUY)
        self.assertIsNone(visible)
        self.assertEqual(len(engine.steps), 6)
        self.assertEqual(engine.expirations, 2)

    def test_new_turn_gets_new_budget_while_old_turn_cannot_reappear(self):
        engine = ToyRecruitEngine(
            observation(remaining=1500),
            after_steps=[observation(remaining=1500), observation(turn=1)],
            after_expirations=[observation(turn=2)],
        )
        timed = self.wrapper(engine)
        timed.reset(seed=17)
        visible = timed.step(BUY)
        self.assertEqual(visible.turn, 2)
        self.assertEqual(visible.action_budget["actions_used"], 0)
        self.assertEqual(visible.action_budget["remaining_actions"], 3)
        with self.assertRaisesRegex(RuntimeError, "closed or earlier"):
            timed.step(BUY)

    def test_unaffordable_pending_choice_uses_adapter_timeout_without_policy_choice(self):
        engine = ToyRecruitEngine(observation(remaining=2500, actions=(CHOOSE,)))
        timed = self.wrapper(engine)
        self.assertIsNone(timed.reset(seed=17))
        self.assertEqual(engine.steps, [])
        self.assertEqual(engine.expirations, 1)
        self.assertEqual(timed.trace[0]["event"], "timeout_or_no_affordable_actions")

    def test_legal_mask_hides_expensive_choices_but_preserves_affordable_ones(self):
        engine = ToyRecruitEngine(observation(remaining=2500, actions=(BUY, CHOOSE, END)))
        timed = self.wrapper(engine)
        visible = timed.reset(seed=17)
        self.assertEqual(visible.legal_actions, (BUY, END))
        with self.assertRaisesRegex(ValueError, "time-constrained legal"):
            timed.step(CHOOSE)
        self.assertEqual(engine.steps, [])

    def test_end_turn_is_free_and_does_not_need_a_profile_cost(self):
        engine = ToyRecruitEngine(observation(remaining=1500))
        timed = self.wrapper(engine)
        timed.reset(seed=17)
        self.assertIsNone(timed.step(END))
        self.assertEqual(engine.steps, [END])
        self.assertEqual(engine.expirations, 0)
        self.assertEqual(timed.trace[0]["before"], timed.trace[0]["after"])

    def test_shorter_live_timer_expires_without_using_initial_action_allowance(self):
        engine = ToyRecruitEngine(
            observation(remaining=10500),
            after_steps=[observation(remaining=1499)],
        )
        timed = self.wrapper(engine)
        timed.reset(seed=17)
        self.assertIsNone(timed.step(BUY))
        self.assertEqual(engine.steps, [BUY])
        self.assertEqual(engine.expirations, 1)
        self.assertEqual(timed.trace[-1]["action_limit"], 10)
        self.assertEqual(timed.trace[-1]["remaining_ms"], 999)

    def test_missing_actual_timer_blocks_even_if_end_turn_exists(self):
        for timer in (None, -1, float("nan"), True):
            with self.subTest(timer=timer):
                engine = ToyRecruitEngine(observation(remaining=timer))
                with self.assertRaises(ValueError):
                    self.wrapper(engine).reset(seed=17)
                self.assertEqual(engine.steps, [])
                self.assertEqual(engine.expirations, 0)

    def test_missing_timing_or_timeout_conformance_blocks_wrapper(self):
        for missing in ("actual_player_window_validated", "pending_choice_timeout_validated"):
            for value in (None, False, "true", 1):
                with self.subTest(missing=missing, value=value):
                    engine = ToyRecruitEngine(observation())
                    engine.report["turn_timing"][missing] = value
                    with self.assertRaises(RecruitCoverageError):
                        self.wrapper(engine)
        engine = ToyRecruitEngine(observation())
        del engine.report["turn_timing"]
        with self.assertRaises(RecruitCoverageError):
            self.wrapper(engine)

    def test_timing_wrapper_does_not_bypass_full_game_coverage_blockers(self):
        engine = ToyRecruitEngine(observation())
        engine.report["blockers"] = ["synthetic-missing-effect"]
        with self.assertRaises(RecruitCoverageError):
            self.wrapper(engine)

    def test_unknown_action_kind_fails_before_engine_can_mutate(self):
        engine = ToyRecruitEngine(observation(actions=(RecruitAction("unmapped"), END)))
        with self.assertRaisesRegex(ValueError, "Unmapped action kind"):
            self.wrapper(engine).reset(seed=17)
        self.assertEqual(engine.steps, [])
        self.assertEqual(engine.expirations, 0)

    def test_expiry_cannot_return_the_same_turn_with_a_fresh_timer(self):
        engine = ToyRecruitEngine(
            observation(remaining=0),
            after_expirations=[observation(remaining=999999)],
        )
        with self.assertRaisesRegex(RuntimeError, "closed or earlier"):
            self.wrapper(engine).reset(seed=17)
        self.assertEqual(engine.expirations, 1)

    def test_adapter_cannot_return_same_turn_after_explicit_end_turn(self):
        engine = ToyRecruitEngine(observation(), after_steps=[observation()])
        timed = self.wrapper(engine)
        timed.reset(seed=17)
        with self.assertRaisesRegex(RuntimeError, "closed or earlier"):
            timed.step(END)
        self.assertEqual(engine.steps, [END])

    def test_unbounded_chain_of_zero_time_turns_fails_closed(self):
        engine = ToyRecruitEngine(
            observation(remaining=0),
            after_expirations=[observation(turn=turn, remaining=0)
                               for turn in range(2, 100)],
        )
        with self.assertRaisesRegex(RuntimeError, "64 consecutive"):
            self.wrapper(engine).reset(seed=17)
        self.assertEqual(engine.expirations, 64)


if __name__ == "__main__":
    unittest.main()
