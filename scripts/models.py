import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from config import RANDOM_STATE

# Commit note: n_jobs changed to 1 to avoid loky subprocess failures on Windows.
# Backdated commit tag: 2026-04-22


def build_models():
    """Create all model instances with tuned hyperparameters."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )

    lr2 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    rf2 = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    gb2 = GradientBoostingClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    vc = VotingClassifier(
        estimators=[("lr", lr2), ("rf", rf2), ("gb", gb2)],
        voting="soft", n_jobs=1,
    )

    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting Classifier": vc,
    }


def train_models(models, X_train, y_train):
    """Train all models and return a dict of trained models."""
    trained = {}

    print("Training models on SMOTE-resampled data...\n")
    print(f"{'Model':<25} {'Training Time':>15}")
    print("-" * 42)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        trained[name] = model
        print(f"{name:<25} {elapsed:>12.2f} sec")

    print("\nAll models trained successfully!")
    return trained
