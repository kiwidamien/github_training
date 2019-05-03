import numpy as np
import pandas as pd


def get_numpy(obj):
    if type(obj) == pd.core.frame.Series:
        return obj.values
    if type(obj) == pd.core.frame.DataFrame:
        return obj.values
    return obj


def get_bootstrap_coefs(scaler, model, features, targets, n_estimators=1000,
                       return_scaled=True):
    coefficents = np.zeros((n_estimators, features.shape[1]))
    features = get_numpy(features)
    targets = get_numpy(targets)

    for row in range(n_estimators):
        sample_mask = np.random.randint(0, len(features),
                                        size=(n_estimators,))
        X_sample = features[sample_mask,]
        y_sample = targets[sample_mask]

        X_sample_scaled = scaler.fit_transform(X_sample)
        model.fit(X_sample_scaled, y_sample)

        coefficents[row, :] = model.coef_
        if not return_scaled:
            coefficents[row, :] = coefficents[row, :]/scaler.scale_
    return coefficents
