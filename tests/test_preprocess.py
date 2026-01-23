"""
Unit tests for stockpreprocess.py

Tests data preprocessing, dataset creation, and dataloader functionality.
"""

import unittest
import sys
import os
import pandas as pd

# Add parent directory to path to import source modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source import stockpreprocess


class TestStockPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions"""

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests - expensive operation"""
        print("\n=== Loading data for preprocessing tests ===")
        cls.stockdf = pd.read_csv(
            cls.test_path
        )  # stockdataloader.load_stock_data(self.test_path)
        (
            cls.train_context,
            cls.train_target,
            cls.val_context,
            cls.val_target,
            cls.train_loader,
            cls.val_loader,
        ) = stockpreprocess.preprocess_stock_data(cls.stockdf)


def suite():
    """Create a test suite combining all test cases"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestStockPreprocessing))
    test_suite.addTest(unittest.makeSuite(TestStockDataset))
    test_suite.addTest(unittest.makeSuite(TestDataLoaders))
    test_suite.addTest(unittest.makeSuite(TestPreprocessingEdgeCases))
    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
