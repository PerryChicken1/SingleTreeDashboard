"""Offline WLS integration checks; no real credentials or licence requests."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from io import StringIO
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

# Also runnable from the development repository.
ROOT = Path(__file__).resolve().parent
if (ROOT.parent / "SingleTreeOptTool").is_dir():
    sys.path.insert(0, str(ROOT.parent / "SingleTreeOptTool"))

import gurobipy as gp
import numpy as np
import pandas as pd
from GurobiLicense import (
    GurobiLicenseError, current_gurobi_environment, gurobi_wls_session, wls_credentials,
)

ACCESS = "11111111-1111-4111-8111-111111111111"
SECRET = "22222222-2222-4222-8222-222222222222"


class FakeEnv:
    instances = []
    fail_start = False

    def __init__(self, *, empty):
        assert empty
        self.params = {}
        self.closed = False
        self.instances.append(self)

    def setParam(self, name, value):
        self.params[name] = value

    def start(self):
        assert list(self.params)[:3] == ["OutputFlag", "LogToConsole", "LogFile"]
        assert self.params["OutputFlag"] == 0 and self.params["LogFile"] == ""
        if self.fail_start:
            raise gp.GurobiError(10009, SECRET)

    def dispose(self):
        self.closed = True


class FakeVector(np.ndarray):
    @property
    def X(self):
        return np.asarray(self)


class FakeModel:
    instances = []
    fail_solve = False
    Status = gp.GRB.OPTIMAL
    SolCount = 1

    def __init__(self, name, *, env):
        assert env is not None, "WLS solve must never use the default environment"
        self.env = env
        self.Params = SimpleNamespace()
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def addMVar(self, count, **kwargs):
        result = np.zeros(count).view(FakeVector)
        result[0] = 1
        return result

    def addConstr(self, *args, **kwargs):
        pass

    def setObjective(self, *args):
        pass

    def optimize(self):
        if self.fail_solve:
            raise gp.GurobiError(10009, SECRET)


class WlsTests(unittest.TestCase):
    def setUp(self):
        FakeEnv.instances = []
        FakeEnv.fail_start = False
        FakeModel.instances = []
        FakeModel.fail_solve = False
        self.credentials = wls_credentials(ACCESS, SECRET, "12345")
        self.env_patch = patch.object(gp, "Env", FakeEnv)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def tearDown(self):
        self.assertIsNone(current_gurobi_environment())

    def test_invalid_inputs_are_rejected_without_echoing_values(self):
        for fields in ((ACCESS, "", "1"), ("private-invalid", SECRET, "1"),
                       (ACCESS, SECRET, "-1"), (ACCESS, SECRET, "1.5")):
            with self.assertRaises(GurobiLicenseError) as raised:
                wls_credentials(*fields)
            self.assertNotIn(SECRET, str(raised.exception))
            self.assertNotIn("private-invalid", str(raised.exception))
        self.assertEqual(FakeEnv.instances, [])

    def test_setup_is_quiet_and_context_is_cleared(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            with gurobi_wls_session(self.credentials):
                env = current_gurobi_environment()
                self.assertEqual(env.params["LICENSEID"], 12345)
                self.assertEqual(env.params["WLSSECRET"], SECRET)
                self.assertIsNone(Context().run(current_gurobi_environment))
        self.assertTrue(env.closed)
        self.assertEqual(output.getvalue(), "")

    def test_authentication_failure_closes_environment_and_hides_secret(self):
        FakeEnv.fail_start = True
        with self.assertRaises(GurobiLicenseError) as raised:
            with gurobi_wls_session(self.credentials):
                self.fail("An unauthenticated run must not execute")
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertTrue(FakeEnv.instances[0].closed)

    def test_unrelated_exception_also_cleans_up(self):
        with self.assertRaisesRegex(ValueError, "test failure"):
            with gurobi_wls_session(self.credentials):
                raise ValueError("test failure")
        self.assertTrue(FakeEnv.instances[0].closed)

    def test_simultaneous_users_get_separate_environments(self):
        barrier = Barrier(2)

        def run(license_id):
            with gurobi_wls_session(wls_credentials(ACCESS, SECRET, str(license_id))):
                env = current_gurobi_environment()
                barrier.wait(timeout=10)
                self.assertEqual(current_gurobi_environment().params["LICENSEID"], license_id)
                return env

        with ThreadPoolExecutor(max_workers=2) as executor:
            environments = list(executor.map(run, [123, 456]))
        self.assertIsNot(environments[0], environments[1])
        self.assertTrue(all(env.closed for env in environments))

    def test_main_and_auxiliary_solves_share_environment_without_retaining_it(self):
        from Constraints import ProportionConstraint
        from Objectives import MinDBHObjective, MinVolumeObjective
        from Optimiser import FutureCropTreeOptimiser
        from SingleTreeAlgorithms import LinearProgrammingAlgorithm
        from SingleTreeDataset import SingleTreeDataset

        dataset = SingleTreeDataset(pd.DataFrame({
            "dbh": [10., 20., 30.], "volume": [1., 2., 3.], "species": ["pine"] * 3,
        }))
        dataset.set_columns(dbh="dbh", volume="volume")
        optimiser = FutureCropTreeOptimiser(
            dataset, {MinDBHObjective(): .5, MinVolumeObjective(): .5},
            [ProportionConstraint(lb=1/3, ub=1/3, prop_col="species", prop_vals={"pine"})],
            LinearProgrammingAlgorithm(),
        )
        with patch.object(gp, "Model", FakeModel):
            with gurobi_wls_session(self.credentials):
                result = optimiser.optimise({"solver_name": "GUROBI"})
        self.assertEqual(len(FakeModel.instances), 5)
        self.assertTrue(all(model.env is FakeEnv.instances[0] for model in FakeModel.instances))
        self.assertTrue(all(model.closed for model in FakeModel.instances))
        self.assertEqual(result["decision_vector"].tolist(), [1, 0, 0])
        self.assertNotIn(SECRET, repr(result))
        self.assertNotIn(SECRET, repr(vars(optimiser.algorithm)))
        self.assertNotIn(FakeEnv.instances[0], vars(optimiser.algorithm).values())

        FakeModel.fail_solve = True
        with patch.object(gp, "Model", FakeModel):
            with self.assertRaises(GurobiLicenseError) as raised:
                with gurobi_wls_session(self.credentials):
                    optimiser.optimise({"solver_name": "GUROBI"})
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertTrue(FakeModel.instances[-1].closed)
        self.assertTrue(FakeEnv.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
