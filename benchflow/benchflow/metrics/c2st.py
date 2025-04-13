from typing import Optional, Union

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X1=None,
    X2=None,
    posterior=None,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    *args,
    **kwargs,
) -> Union[np.ndarray, str]:
    """Classifier-based 2-sample test returning accuracy.

    Trains classifiers with N-fold cross-validation [1]. Scikit-learn
    MLPClassifier are used, with 2 hidden layers of 10x dim each, where dim is
    the dimensionality of the samples X1 and Y1.

    Parameters
    ----------
    X1 : np.ndarray, optional
        A ``N1``-by-``D`` matrix of samples.
    X2 : np.ndarray, optional
        Another ``N2``-by-``D`` matrix of samples.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.
    seed : int, optional
        The random seed, by default 1.
    n_folds : int, optional
        Number of folds. Must be at least 2. By default 5.
    scoring : str, optional
        Scoring method, by default "accuracy"
    z_score : bool, optional
        Z-scoring using X1, by default True
    noise_scale : Optional[float], optional
        If passed, will add Gaussian noise with std ``noise_scale`` to samples, by
        default None.

    Returns
    -------
    np.ndarray
        Accuracy.

    References
    ----------
    [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples or posterior provided.")
    if posterior is not None:
        X1 = posterior.get_samples()
        X2 = posterior.task.get_posterior_samples()
        if isinstance(X1, Exception):  # Record errors, if any
            return X1
        if isinstance(X2, Exception):
            return X2

    if z_score:
        X_mean = np.mean(X1, axis=0)
        X_std = np.std(X1, axis=0, ddof=1)
        X1 = (X1 - X_mean) / X_std
        X2 = (X2 - X_mean) / X_std

    if noise_scale is not None:
        X1 += noise_scale * np.random.randn(X1.shape)
        X2 += noise_scale * np.random.randn(X2.shape)

    ndim = X1.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=200,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X1, X2))
    target = np.concatenate(
        (
            np.zeros((X1.shape[0],)),
            np.ones((X2.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return np.atleast_1d(scores)


def c2st_auc(
    X1: Optional[np.ndarray] = None,
    X2: Optional[np.ndarray] = None,
    posterior=None,
    seed: int = 1,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> Union[np.ndarray, str]:
    """Classifier-based 2-sample test returning AUC (area under curve).

    Same as c2st, except that it returns ROC AUC rather than accuracy.

    Returns
    -------
    np.ndarray
        ROC AUC.
    """
    return c2st(
        X1,
        X2,
        posterior,
        seed=seed,
        n_folds=n_folds,
        scoring="roc_auc",
        z_score=z_score,
        noise_scale=noise_scale,
    )
