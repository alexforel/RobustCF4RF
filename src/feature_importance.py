from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def feature_permutation_importance(x, y, nbRepeats, maxDepth, m_n):
    importance = []
    for i in range(nbRepeats):
        # Create and train random forest classifier
        rfClassifier = RandomForestClassifier(n_estimators=m_n,
                                              max_depth=maxDepth,
                                              max_features="sqrt")
        rfClassifier.fit(x, y)
        # Split data in train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, stratify=y, random_state=42)
        result = permutation_importance(rfClassifier, X_test, y_test,
                                        n_repeats=8, random_state=42, n_jobs=4)
        importance.append(result.importances)
    return importance
