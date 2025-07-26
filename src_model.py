from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    return report, auc

def save_model(clf, scaler, path="model.joblib"):
    joblib.dump({'model': clf, 'scaler': scaler}, path)