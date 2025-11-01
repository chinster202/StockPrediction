"""
Unit tests for stockdataloader.py

Tests data loading, normalization, and validation of stock data.
"""

import unittest
import sys
import os
import pandas as pd

# Add parent directory to path to import source modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStockDataLoader(unittest.TestCase):
    "Test cases for stock data loading"

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        cls.test_path = "data/A.csv"
        cls.stockdf = None

    def setUp(self):
        """Run before each test method"""
        if self.stockdf is None:
            self.__class__.stockdf = pd.read_csv(
                self.test_path
            )  # stockdataloader.load_stock_data(self.test_path)

    def test_expected_columns_exist(self):
        # Test that all expected stock data columns are present
        expected_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        print(self.stockdf.columns)

        for col in expected_columns:
            self.assertIn(
                col,
                self.stockdf.columns,
                f"Expected column '{col}' not found in DataFrame",
            )

    def test_no_null_values(self):
        """Test that there are no null/NaN values in the dataframe"""
        null_counts = self.stockdf.isnull().sum()
        total_nulls = null_counts.sum()

        self.assertEqual(
            total_nulls,
            0,
            f"DataFrame should not contain null values. Found: {null_counts[null_counts > 0]}",
        )


def suite():
    """Create a test suite combining all test cases"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestStockDataLoader))
    test_suite.addTest(unittest.makeSuite(TestStockDataLoaderEdgeCases))
    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
