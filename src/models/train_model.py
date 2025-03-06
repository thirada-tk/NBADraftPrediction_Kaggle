from sklearn.metrics import roc_auc_score

def fit_assess_classifier(model, X_train, y_train, X_val, y_val):
    """Train a classifier model, print its AUROC score on the training and validation set, and return the trained model.

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn classifier model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_val : Numpy Array
        Features for the validation set
    y_val : Numpy Array
        Target for the validation set

    Returns
    -------
    sklearn.base.BaseEstimator
        Trained classifier model
    """
    model.fit(X_train, y_train)

    # Predict probability of positive class for training and validation sets
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate AUROC for training and validation sets
    auroc_train = roc_auc_score(y_train, y_train_pred_proba)
    auroc_val = roc_auc_score(y_val, y_val_pred_proba)

    # Print AUROC scores
    print(f"Training AUROC Score: {auroc_train:.2f}")
    print(f"Validation AUROC Score: {auroc_val:.2f}")

    return model


