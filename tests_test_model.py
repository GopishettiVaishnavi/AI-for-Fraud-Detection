import unittest
from src.preprocess import load_and_preprocess
from src.model import train_model, evaluate_model

class TestFraudModel(unittest.TestCase):
    def test_train_and_evaluate(self):
        X_train, X_test, y_train, y_test, _ = load_and_preprocess("data/creditcard.csv")
        clf = train_model(X_train, y_train)
        report, auc = evaluate_model(clf, X_test, y_test)
        self.assertGreater(auc, 0.80)

if __name__ == "__main__":
    unittest.main()