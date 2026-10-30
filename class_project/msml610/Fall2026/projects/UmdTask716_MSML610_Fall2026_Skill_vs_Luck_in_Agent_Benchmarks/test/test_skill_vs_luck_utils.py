"""
Import as:

import test.test_skill_vs_luck_utils as ttsvlut
"""

import logging
import pprint
from typing import Tuple

import numpy as np
import pandas as pd

import helpers.hunit_test as hunitest

import skill_vs_luck_utils as utils

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_parse_results_json
# #############################################################################


class Test_parse_results_json(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_results_json()`.
    """

    def test1(self) -> None:
        """
        Test that the solved tasks become 1 and the other tasks 0.
        """
        # Prepare inputs.
        results = {"resolved": ["task_b"], "no_generation": ["task_c"]}
        task_ids = ["task_a", "task_b", "task_c"]
        # Prepare outputs.
        expected = "{'task_a': 0, 'task_b': 1, 'task_c': 0}"
        # Run test.
        outcomes = utils.parse_results_json(results, task_ids)
        # Check outputs.
        actual = str(outcomes.to_dict())
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test a submission that solved no task.
        """
        # Prepare inputs.
        results = {}
        task_ids = ["task_a", "task_b"]
        # Prepare outputs.
        expected = "{'task_a': 0, 'task_b': 0}"
        # Run test.
        outcomes = utils.parse_results_json(results, task_ids)
        # Check outputs.
        actual = str(outcomes.to_dict())
        self.assert_equal(actual, expected)


# #############################################################################
# Test_parse_per_instance_details
# #############################################################################


class Test_parse_per_instance_details(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_per_instance_details()`.
    """

    def test1(self) -> None:
        """
        Test that the records follow the order of the task ids.
        """
        # Prepare inputs.
        details = {
            "task_b": {"cost": 0.5, "api_calls": 30, "resolved": False},
            "task_a": {"cost": 0.25, "api_calls": 12, "resolved": True},
        }
        task_ids = ["task_a", "task_b"]
        # Prepare outputs.
        expected = """
                resolved  cost  api_calls
        task_a         1  0.25         12
        task_b         0  0.50         30
        """
        # Run test.
        records = utils.parse_per_instance_details(details, task_ids)
        # Check outputs.
        actual = records.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_parse_metadata
# #############################################################################


class Test_parse_metadata(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_metadata()`.
    """

    def test1(self) -> None:
        """
        Test a full `metadata.yaml` where some fields are lists.
        """
        # Prepare inputs.
        metadata = {
            "info": {"name": "GPT 5 mini", "resolved": 56.2},
            "tags": {
                "agent": "mini-SWE-agent",
                "model": ["gpt-5-mini-2025-08-07"],
                "org": ["OpenAI"],
                "system": {"attempts": 1},
            },
        }
        folder = "20260217_mini-v2.0.0_gpt-5-mini"
        # Prepare outputs.
        expected = """
        {'agent': 'mini-SWE-agent',
         'attempts': '1',
         'date': '2026-02-17',
         'model': 'gpt-5-mini-2025-08-07',
         'name': 'GPT 5 mini',
         'org': 'OpenAI',
         'reported_score': 56.2,
         'submission': '20260217_mini-v2.0.0_gpt-5-mini'}
        """
        # Run test.
        fields = utils.parse_metadata(metadata, folder)
        # Check outputs.
        actual = pprint.pformat(fields)
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test a submission without `metadata.yaml`.
        """
        # Prepare inputs.
        metadata = {}
        folder = "20250728_zai_glm4-5"
        # Prepare outputs.
        expected = """
        {'agent': '',
         'attempts': '',
         'date': '2025-07-28',
         'model': '',
         'name': '20250728_zai_glm4-5',
         'org': '',
         'reported_score': nan,
         'submission': '20250728_zai_glm4-5'}
        """
        # Run test.
        fields = utils.parse_metadata(metadata, folder)
        # Check outputs.
        actual = pprint.pformat(fields)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_leaderboard
# #############################################################################


class Test_compute_leaderboard(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_leaderboard()`.
    """

    def test1(self) -> None:
        """
        Test the ranking, a tie, and 2 agents with the same name.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame(
            [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0]],
            index=["sub_a", "sub_b", "sub_c"],
            columns=["t1", "t2", "t3", "t4"],
        )
        submissions = pd.DataFrame(
            {
                "name": ["Model X", "Model Y", "Model X"],
                "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
                "agent": ["", "", ""],
                "attempts": ["1", "1", "2+"],
            },
            index=["sub_a", "sub_b", "sub_c"],
        )
        # Prepare outputs.
        expected = """
               rank                 label  score  n_resolved     name        date agent attempts
        sub_b     1               Model Y   0.75           3  Model Y  2025-02-01              1
        sub_a     2  Model X (2025-01-01)   0.50           2  Model X  2025-01-01              1
        sub_c     2  Model X (2025-03-01)   0.50           2  Model X  2025-03-01             2+
        """
        # Run test.
        leaderboard = utils.compute_leaderboard(outcomes, submissions)
        # Check outputs.
        actual = leaderboard.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_score_cis
# #############################################################################


class Test_compute_score_cis(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_score_cis()`.
    """

    def test1(self) -> None:
        """
        Test the scores, the binomial standard errors, and that each interval
        contains the score.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame(
            [[1, 1, 1, 0] * 25, [0, 1, 0, 0] * 25],
            index=["sub_a", "sub_b"],
        )
        # Prepare outputs.
        expected = """
               score  se_binomial  contains_score
        sub_a   0.75       0.0433            True
        sub_b   0.25       0.0433            True
        """
        # Run test.
        cis = utils.compute_score_cis(outcomes, n_boot=500, seed=1)
        # Check outputs.
        cis["contains_score"] = (cis["ci_low"] <= cis["score"]) & (
            cis["score"] <= cis["ci_high"]
        )
        actual = cis[["score", "se_binomial", "contains_score"]].round(4)
        actual = actual.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compare_pair
# #############################################################################


class Test_compare_pair(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compare_pair()`.
    """

    def test1(self) -> None:
        """
        Test 2 agents that disagree on 2 tasks, both solved only by A.
        """
        # Prepare inputs.
        outcomes_a = pd.Series([1, 1, 1, 0])
        outcomes_b = pd.Series([0, 0, 1, 0])
        # Prepare outputs.
        # A wins both disagreements, so the exact McNemar p-value is
        # 2 * 0.5^2 = 0.5.
        expected = "{'gap': 0.5, 'only_a': 2, 'only_b': 0, 'p_value': 0.5, 'sd_diff': 0.5774}"
        # Run test.
        comparison = utils.compare_pair(outcomes_a, outcomes_b, n_boot=200)
        # Check outputs.
        keys = ["gap", "only_a", "only_b", "p_value", "sd_diff"]
        actual = {key: round(comparison[key], 4) for key in keys}
        actual = pprint.pformat(actual)
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test 2 agents with identical outcomes.
        """
        # Prepare inputs.
        outcomes_a = pd.Series([1, 0, 1, 0])
        outcomes_b = pd.Series([1, 0, 1, 0])
        # Prepare outputs.
        expected = """
        {'ci_high': 0.0,
         'ci_low': 0.0,
         'gap': 0.0,
         'only_a': 0,
         'only_b': 0,
         'p_value': 1.0,
         'sd_diff': 0.0}
        """
        # Run test.
        comparison = utils.compare_pair(outcomes_a, outcomes_b, n_boot=200)
        # Check outputs.
        actual = pprint.pformat(comparison)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_holm_adjust
# #############################################################################


class Test_holm_adjust(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.holm_adjust()`.
    """

    def test1(self) -> None:
        """
        Test 3 p-values where 2 adjusted values end up equal.
        """
        # Prepare inputs.
        p_values = [0.01, 0.04, 0.03]
        # Prepare outputs.
        expected = "[0.03, 0.06, 0.06]"
        # Run test.
        adjusted = utils.holm_adjust(p_values)
        # Check outputs.
        actual = str([round(float(value), 4) for value in adjusted])
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that adjusted p-values never exceed 1.
        """
        # Prepare inputs.
        p_values = [0.6, 0.7]
        # Prepare outputs.
        expected = "[1.0, 1.0]"
        # Run test.
        adjusted = utils.holm_adjust(p_values)
        # Check outputs.
        actual = str([round(float(value), 4) for value in adjusted])
        self.assert_equal(actual, expected)


# #############################################################################
# Test_find_records
# #############################################################################


def _get_record_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build 3 submissions on 40 tasks where each one sets a record.

    The second record solves 2 more tasks than the first, too few to tell
    them apart, and the third solves 18 more than the second.

    :return: outcome matrix, the records from `find_records()`, and the
        audit from `audit_records()`
    """
    outcomes = pd.DataFrame(
        [
            [1] * 10 + [0] * 30,
            [1] * 12 + [0] * 28,
            [1] * 30 + [0] * 10,
        ],
        index=["sub_a", "sub_b", "sub_c"],
    )
    submissions = pd.DataFrame(
        {
            "name": ["Agent A", "Agent B", "Agent C"],
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "agent": ["", "", ""],
            "attempts": ["1", "1", "2+"],
        },
        index=outcomes.index,
    )
    leaderboard = utils.compute_leaderboard(outcomes, submissions)
    records = utils.find_records(leaderboard)
    audit = utils.audit_records(outcomes, records, n_boot=200)
    return outcomes, records, audit


class Test_find_records(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.find_records()`.
    """

    def test1(self) -> None:
        """
        Test a tie with the record, a tie on the same day, and a day with a
        weaker second submission.
        """
        # Prepare inputs: `sub_c` only ties the record of `sub_a`, and `sub_d`
        # and `sub_e` tie on the same day, so the first folder name wins.
        outcomes = pd.DataFrame(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            index=["sub_a", "sub_b", "sub_c", "sub_d", "sub_e", "sub_f"],
        )
        submissions = pd.DataFrame(
            {
                "name": ["A", "B", "C", "D", "E", "F"],
                "date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-03-01",
                    "2024-04-01",
                ],
                "agent": [""] * 6,
                "attempts": ["1"] * 6,
            },
            index=outcomes.index,
        )
        leaderboard = utils.compute_leaderboard(outcomes, submissions)
        # Prepare outputs.
        expected = """
              previous        date  n_resolved
        sub_a           2024-01-01           2
        sub_d    sub_a  2024-03-01           3
        sub_f    sub_d  2024-04-01           4
        """
        # Run test.
        records = utils.find_records(leaderboard)
        # Check outputs.
        actual = records[["previous", "date", "n_resolved"]].to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_audit_records
# #############################################################################


class Test_audit_records(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.audit_records()`.
    """

    def test1(self) -> None:
        """
        Test a record too close to the one it broke and a record far above.
        """
        # Prepare inputs.
        _, _, audit = _get_record_inputs()
        # Prepare outputs.
        # `sub_b` wins its 2 disagreements with `sub_a`, so the exact McNemar
        # p-value is 2 * 0.5^2 = 0.5, and `sub_c` wins 18 of 18, so it is
        # 2 * 0.5^18 = 7.6e-06, doubled by Holm.
        expected = """
                record previous   gap  only_a  only_b   p_value    p_holm  distinguishable
        sub_b  Agent B  Agent A  0.05       2       0  0.500000  0.500000            False
        sub_c  Agent C  Agent B  0.45      18       0  0.000008  0.000015             True
        """
        # Run test.
        columns = ["record", "previous", "gap", "only_a", "only_b"]
        columns += ["p_value", "p_holm", "distinguishable"]
        actual = audit[columns].to_string()
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_summarize_records
# #############################################################################


class Test_summarize_records(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.summarize_records()`.
    """

    def test1(self) -> None:
        """
        Test the counts and the share of the progress from distinguishable
        steps.
        """
        # Prepare inputs.
        outcomes, records, audit = _get_record_inputs()
        # Prepare outputs.
        # The 2 steps add 2 and 18 tasks out of 40 and only the second is
        # distinguishable, so it carries 18 / 20 of the progress. The last
        # record wins 20 of 20 disagreements with the first: p = 2 * 0.5^20.
        expected = """
        {'distinguishable_share': 0.9,
         'n_distinguishable': 1,
         'n_multi_attempt': 1,
         'n_records': 2,
         'p_first_last': 1.91e-06,
         'total_gap': 0.5}
        """
        # Run test.
        summary = utils.summarize_records(outcomes, records, audit)
        # Check outputs.
        actual = {key: round(value, 8) for key, value in summary.items()}
        actual = pprint.pformat(actual)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_check_record_predictions
# #############################################################################


class Test_check_record_predictions(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.check_record_predictions()`.
    """

    def test1(self) -> None:
        """
        Test that exactly half of the records being distinguishable misses A1,
        while A2 and A3 hold.
        """
        # Prepare inputs.
        outcomes, records, audit = _get_record_inputs()
        summary = utils.summarize_records(outcomes, records, audit)
        # Prepare outputs.
        expected = """
        A1: False, 1 of 2 records distinguishable
        A2: True, gap of 50.0 points, exact McNemar p 1.91e-06
        A3: True, 90% of the points
        """
        # Run test.
        table = utils.check_record_predictions(summary)
        # Check outputs.
        actual = "\n".join(
            f"{name}: {row['holds']}, {row['observed']}"
            for name, row in table.iterrows()
        )
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_find_distinct_levels
# #############################################################################


class Test_find_distinct_levels(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.find_distinct_levels()`.
    """

    def test1(self) -> None:
        """
        Test that a close record stays in the level of the first one and a far
        one starts a new level.
        """
        # Prepare inputs.
        outcomes, records, _ = _get_record_inputs()
        # Prepare outputs.
        # `sub_b` is 2 tasks above `sub_a` (p = 0.5), so it joins its level,
        # and `sub_c` is 20 tasks above `sub_a` (p = 2 * 0.5^20).
        expected = """
               n_records  gap   p_value
        sub_a          2  NaN       NaN
        sub_c          1  0.5  0.000002
        """
        # Run test.
        levels = utils.find_distinct_levels(outcomes, records)
        # Check outputs.
        actual = levels[["n_records", "gap", "p_value"]].round(6).to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_min_tasks
# #############################################################################


class Test_compute_min_tasks(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_min_tasks()`.
    """

    def test1(self) -> None:
        """
        Test the default 80% power.
        """
        # Prepare inputs.
        sd_diff = 0.5
        gap = 0.05
        # Prepare outputs.
        # (1.96 + 0.8416)^2 * 0.5^2 / 0.05^2 = 784.89.
        expected = 784.89
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap)
        # Check outputs.
        self.assertAlmostEqual(n_tasks, expected, places=2)

    def test2(self) -> None:
        """
        Test 50% power, which matches the formula in the spec.
        """
        # Prepare inputs.
        sd_diff = 0.5
        gap = 0.05
        # Prepare outputs.
        # 1.96^2 * 0.5^2 / 0.05^2 = 384.15.
        expected = 384.15
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap, power=0.5)
        # Check outputs.
        self.assertAlmostEqual(n_tasks, expected, places=2)


# #############################################################################
# Test_compute_min_detectable_gap
# #############################################################################


class Test_compute_min_detectable_gap(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_min_detectable_gap()`.
    """

    def test1(self) -> None:
        """
        Test a benchmark of 400 tasks at the default 80% power.
        """
        # Prepare inputs.
        sd_diff = 0.5
        n_tasks = 400
        # Prepare outputs.
        # (1.96 + 0.8416) * 0.5 / sqrt(400) = 0.0700.
        expected = 0.0700
        # Run test.
        gap = utils.compute_min_detectable_gap(sd_diff, n_tasks)
        # Check outputs.
        self.assertAlmostEqual(gap, expected, places=4)

    def test2(self) -> None:
        """
        Test that it inverts `compute_min_tasks()`.
        """
        # Prepare inputs.
        sd_diff = 0.43
        gap = 0.03
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap)
        actual = utils.compute_min_detectable_gap(sd_diff, n_tasks)
        # Check outputs.
        self.assertAlmostEqual(actual, gap, places=10)


# #############################################################################
# Test_compute_expected_scores
# #############################################################################


class Test_compute_expected_scores(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_expected_scores()`.
    """

    def test1(self) -> None:
        """
        Test an agent with ability 0, whose score is 0.5 by symmetry.
        """
        # Prepare inputs.
        abilities = [0.0]
        # Prepare outputs.
        expected = 0.5
        # Run test.
        scores = utils.compute_expected_scores(abilities, difficulty_sd=1.5)
        # Check outputs.
        self.assertAlmostEqual(scores["agent_0"], expected, places=6)

    def test2(self) -> None:
        """
        Test tasks that are all equally hard, so the score is the sigmoid of
        the ability.
        """
        # Prepare inputs.
        abilities = [1.0]
        # Prepare outputs.
        expected = 1 / (1 + np.exp(-1.0))
        # Run test.
        scores = utils.compute_expected_scores(abilities, difficulty_sd=0.0)
        # Check outputs.
        self.assertAlmostEqual(scores["agent_0"], expected, places=6)


# #############################################################################
# Test_simulate_outcomes
# #############################################################################


class Test_simulate_outcomes(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.simulate_outcomes()`.
    """

    def test1(self) -> None:
        """
        Test the shape and the labels of the simulated matrix.
        """
        # Prepare inputs.
        abilities = [-1.0, 0.0, 1.0]
        # Prepare outputs.
        expected = """
        (3, 4)
        ['agent_0', 'agent_1', 'agent_2']
        ['task_0', 'task_1', 'task_2', 'task_3']
        """
        # Run test.
        outcomes = utils.simulate_outcomes(abilities, 4, seed=3)
        # Check outputs.
        actual = "\n".join(
            [
                str(outcomes.shape),
                str(outcomes.index.tolist()),
                str(outcomes.columns.tolist()),
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that the same seed gives the same benchmark.
        """
        # Prepare inputs.
        abilities = [0.0, 0.5]
        # Run test.
        outcomes_1 = utils.simulate_outcomes(abilities, 50, seed=7)
        outcomes_2 = utils.simulate_outcomes(abilities, 50, seed=7)
        # Check outputs.
        self.assertTrue(outcomes_1.equals(outcomes_2))


# #############################################################################
# Test_fit_pass_chances
# #############################################################################


def _get_irt_inputs() -> pd.DataFrame:
    """
    Build 4 agents x 5 tasks where no agent and no task is all 0 or all 1.
    """
    outcomes = pd.DataFrame(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 0, 1],
        ],
        index=["agent_0", "agent_1", "agent_2", "agent_3"],
        columns=["task_0", "task_1", "task_2", "task_3", "task_4"],
    )
    return outcomes


class Test_fit_pass_chances(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.fit_pass_chances()`.
    """

    def test1(self) -> None:
        """
        Test that with almost no penalty the Rasch fit expects as many passes
        as each agent and each task have.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        # Prepare outputs.
        expected = """
        agents: [3.0, 3.0, 3.0, 2.0]
        tasks: [3.0, 3.0, 2.0, 1.0, 2.0]
        """
        # Run test.
        chances = utils.fit_pass_chances(outcomes, penalty=1e-6)
        # Check outputs.
        actual = "\n".join(
            [
                f"agents: {chances.sum(axis=1).round(3).tolist()}",
                f"tasks: {chances.sum(axis=0).round(3).tolist()}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a task solved by every agent gets the highest chance of each
        agent, still below 1.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        outcomes["task_easy"] = 1
        # Prepare outputs.
        expected = """
        chances: [0.993, 0.993, 0.993, 0.981]
        highest: ['task_easy', 'task_easy', 'task_easy', 'task_easy']
        """
        # Run test.
        chances = utils.fit_pass_chances(outcomes)
        # Check outputs.
        actual = "\n".join(
            [
                f"chances: {chances['task_easy'].round(3).tolist()}",
                f"highest: {chances.idxmax(axis=1).tolist()}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test that agents with the same score get the same chances with Rasch
        but not with 2PL, which also looks at which tasks they solve.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        # Prepare outputs.
        expected = """
        rasch: True
        2pl: False
        """
        # Run test.
        rasch = utils.fit_pass_chances(outcomes)
        two_pl = utils.fit_pass_chances(outcomes, irt_model="2pl")
        # Check outputs.
        # `agent_0` and `agent_1` both solve 3 of the 5 tasks.
        same_rasch = np.allclose(rasch.loc["agent_0"], rasch.loc["agent_1"])
        same_2pl = np.allclose(two_pl.loc["agent_0"], two_pl.loc["agent_1"])
        actual = f"rasch: {same_rasch}\n2pl: {same_2pl}"
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        Test that an unknown model is rejected.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            utils.fit_pass_chances(outcomes, irt_model="3pl")


# #############################################################################
# Test_estimate_run_share
# #############################################################################


class Test_estimate_run_share(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.estimate_run_share()`.
    """

    def test1(self) -> None:
        """
        Test that the estimate is close to the true run share when the agents
        follow the Rasch model.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        abilities = np.linspace(-1.5, 1.5, 30)
        difficulties = rng.normal(0, 1.5, size=400)
        true_chances = 1 / (
            1 + np.exp(difficulties[None, :] - abilities[:, None])
        )
        solved = rng.random(true_chances.shape) < true_chances
        outcomes = pd.DataFrame(solved.astype(int))
        # The true run share of each agent comes from its true chances.
        true_score = true_chances.mean(axis=1)
        truth = (true_chances * (1 - true_chances)).mean(axis=1) / (
            true_score * (1 - true_score)
        )
        # Prepare outputs.
        expected = """
        truth: [0.75, 0.71, 0.74]
        estimate: [0.72, 0.69, 0.7]
        max_error: 0.036
        """
        # Run test.
        estimates = utils.estimate_run_share(outcomes)
        # Check outputs.
        errors = np.abs(estimates["run_share"].to_numpy() - truth)
        agents = [0, 15, 29]
        estimate = estimates["run_share"].iloc[agents].round(2).tolist()
        actual = "\n".join(
            [
                f"truth: {np.round(truth[agents], 2).tolist()}",
                f"estimate: {estimate}",
                f"max_error: {round(float(errors.max()), 3)}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that an agent that solves every task gets no run share.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        outcomes.loc["agent_4"] = 1
        # Prepare outputs.
        expected = """
                 score  sigma2_run  run_share
        agent_0    0.6       0.206      0.857
        agent_1    0.6       0.206      0.857
        agent_2    0.6       0.206      0.857
        agent_3    0.4       0.209      0.869
        agent_4    1.0       0.009        NaN
        """
        # Run test.
        estimates = utils.estimate_run_share(outcomes)
        # Check outputs.
        actual = estimates.round(3).to_string()
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test that the calibration factor scales the run share.
        """
        # Prepare inputs.
        outcomes = _get_irt_inputs()
        # Prepare outputs.
        expected = "[0.5, 0.5, 0.5, 0.5]"
        # Run test.
        base = utils.estimate_run_share(outcomes)
        scaled = utils.estimate_run_share(outcomes, calibration=0.5)
        # Check outputs.
        ratio = scaled["run_share"] / base["run_share"]
        actual = str(ratio.round(6).tolist())
        self.assert_equal(actual, expected)
