import math
import unittest

import numpy as np
import pandas as pd
from numpy import testing as np_test
from pandas.api.types import CategoricalDtype

from pgmpy.estimators.CITests import *

np.random.seed(42)


class TestPearsonr(unittest.TestCase):
    def setUp(self):
        self.df_ind = pd.DataFrame(np.random.randn(10000, 3), columns=["X", "Y", "Z"])

        Z = np.random.randn(10000)
        X = 3 * Z + np.random.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z + np.random.normal(loc=0, scale=0.1, size=10000)

        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        Z1 = np.random.randn(10000)
        Z2 = np.random.randn(10000)
        X = 3 * Z1 + 2 * Z2 + np.random.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z1 + 3 * Z2 + np.random.normal(loc=0, scale=0.1, size=10000)
        self.df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

        X = np.random.rand(10000)
        Y = np.random.rand(10000)
        Z = 2 * X + 2 * Y + np.random.normal(loc=0, scale=0.1, size=10000)
        self.df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_pearsonr(self):
        coef, p_value = pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, boolean=False)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=self.df_cind, boolean=False
        )
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z1", "Z2"], data=self.df_cind_mul, boolean=False
        )
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=self.df_vstruct, boolean=False
        )
        self.assertTrue(abs(coef) > 0.9)
        self.assertTrue(p_value < 0.05)

        # Tests for when boolean=True
        self.assertTrue(
            pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, significance_level=0.05)
        )
        self.assertTrue(
            pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind, significance_level=0.05)
        )
        self.assertTrue(
            pearsonr(
                X="X",
                Y="Y",
                Z=["Z1", "Z2"],
                data=self.df_cind_mul,
                significance_level=0.05,
            )
        )
        self.assertFalse(
            pearsonr(
                X="X", Y="Y", Z=["Z"], data=self.df_vstruct, significance_level=0.05
            )
        )


class TestCITests(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    def test_chisquare_adult_dataset(self):
        # Comparison values taken from dagitty (DAGitty)
        coef, p_value, dof = chi_square(
            X="Age", Y="Immigrant", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -25.47, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Age", Y="Race", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -24.75, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Age", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 289.62, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -139.82, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Education",
            Y="HoursPerWeek",
            Z=["Age", "Immigrant", "Race", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 1460.11, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        self.assertEqual(dof, 316)

        coef, p_value, dof = chi_square(
            X="Immigrant", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 0.2724, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -0.50, decimal=1)
        self.assertEqual(dof, 1)

        coef, p_value, dof = chi_square(
            X="Education",
            Y="MaritalStatus",
            Z=["Age", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 481.96, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        self.assertEqual(dof, 58)

        # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
        # dataframes with very few samples. Update: Might be same from scipy=1.7.0
        coef, p_value, dof = chi_square(
            X="Income",
            Y="Race",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 66.39, decimal=1)
        np_test.assert_almost_equal(p_value, 0.99, decimal=1)
        self.assertEqual(dof, 136)

        coef, p_value, dof = chi_square(
            X="Immigrant",
            Y="Income",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 65.59, decimal=1)
        np_test.assert_almost_equal(p_value, 0.999, decimal=2)
        self.assertEqual(dof, 131)

    def test_discrete_tests(self):
        for t in [
            chi_square,
            g_sq,
            log_likelihood,
            freeman_tuckey,
            modified_log_likelihood,
            neyman,
            cressie_read,
        ]:
            self.assertFalse(
                t(
                    X="Age",
                    Y="Immigrant",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Age",
                    Y="Race",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Age",
                    Y="Sex",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Education",
                    Y="HoursPerWeek",
                    Z=["Age", "Immigrant", "Race", "Sex"],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )
            self.assertTrue(
                t(
                    X="Immigrant",
                    Y="Sex",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )
            self.assertFalse(
                t(
                    X="Education",
                    Y="MaritalStatus",
                    Z=["Age", "Sex"],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

    def test_exactly_same_vars(self):
        x = np.random.choice([0, 1], size=1000)
        y = x.copy()
        df = pd.DataFrame({"x": x, "y": y})

        for t in [
            chi_square,
            g_sq,
            log_likelihood,
            freeman_tuckey,
            modified_log_likelihood,
            neyman,
            cressie_read,
        ]:
            stat, p_value, dof = t(X="x", Y="y", Z=[], data=df, boolean=False)
            self.assertEqual(dof, 1)
            np_test.assert_almost_equal(p_value, 0, decimal=5)


class TestResidual(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=42)
        z = np.absolute(np.random.normal(0, 0.9, 1000))
        x = 0.7 * z + np.random.normal(0, 0.1, 1000)
        y = 0.8 * z + np.random.normal(0, 0.1, 1000)

        # Create dataset with all variables being the same type
        self.df_cont = pd.DataFrame({"X": x, "Y": y, "Z": z})

        cat_ord = CategoricalDtype(categories=range(3), ordered=True)
        self.df_ord = self.df_cont.round().astype(cat_ord).dropna()

        cat_cat = CategoricalDtype(categories=range(3), ordered=False)
        self.df_cat = (self.df_cont / 3).round().astype(int).astype(cat_cat).dropna()

        # Create datasets with mixed data types

    def test_residual_single_data_type(self):
        chi1, p_value1 = residual_test(
            "X", "Y", ["Z"], data=self.df_cont, boolean=False
        )
        np_test.assert_almost_equal(chi1, 0.164, decimal=3)
        np_test.assert_almost_equal(p_value1, 0.685, decimal=3)

        chi2, p_value2 = residual_test("X", "Y", ["Z"], data=self.df_ord, boolean=False)
        np_test.assert_almost_equal(chi2, 114.668, decimal=3)
        np_test.assert_almost_equal(p_value2, 0, decimal=3)

        chi3, p_value3 = residual_test("X", "Y", ["Z"], data=self.df_cat, boolean=False)
        np_test.assert_almost_equal(chi3, 7683.278, decimal=3)
        np_test.assert_almost_equal(p_value3, 0, decimal=3)
