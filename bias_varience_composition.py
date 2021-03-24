import numpy as np

def _draw_bootstrap_sample(rng, X, y, frac = 1.0):
    """
    Function to draw a single bootstrap sub-sample from given data.
    Use frac to adjust the size of the sample as a fraction of data. Defaults to size of given data.
    """
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=int(sample_indices.shape[0] * frac),
                                   replace=True)
    return X[bootstrap_indices], y[bootstrap_indices] 

def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test,
                         num_rounds=200, random_seed=None, bootstrap_frac = 1.0):
    """
    Extended from: mlxtend library (URL:http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)
    
    Function to generate "num_rounds" realizations of the given ML estimator using bootstrap sub-samples.
    """
    
    rng = np.random.RandomState(random_seed)

    # Array with predictions of all test set observations for "num_round" bootstrap sub-samples
    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.int)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train, frac = bootstrap_frac)
        pred = estimator.fit(X_boot, y_boot).predict(X_test)
        all_pred[i] = pred

    avg_expected_loss = np.apply_along_axis(
        lambda x: ((x - y_test)**2).mean(),
        axis=1,
        arr=all_pred).mean()

    main_predictions = np.mean(all_pred, axis=0)

    avg_bias = np.sum((main_predictions - y_test)**2) / y_test.size
    avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var, all_pred