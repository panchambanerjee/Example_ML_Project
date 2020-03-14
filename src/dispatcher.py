from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, verbose=2),
    'extratrees': ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42, verbose=2),
          }