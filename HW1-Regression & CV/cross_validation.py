import numpy as np

from performance_metrics import calc_root_mean_squared_error


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    Examples
    --------
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    >>> N = 101
    >>> n_folds = 7
    >>> x_N3 = np.random.RandomState(0).rand(N, 3)
    >>> y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    >>> y_N.shape
    (101,)

    >>> import sklearn.linear_model
    >>> my_regr = sklearn.linear_model.LinearRegression()
    >>> tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    ...                 my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)

    # Training error should be indistiguishable from zero
    >>> np.array2string(tr_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'

    # Testing error should be indistinguishable from zero
    >>> np.array2string(te_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'
    '''

    assert y_N.ndim == 1
    assert x_NF.ndim == 2

    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float32)

    N, F = x_NF.shape

    # define the folds here by calling your function
    train_ids_per_fold, test_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(
        N, n_folds, random_state)

    # loop over folds and compute the train and test error

    for i in range(n_folds):
        train_set_x = x_NF[train_ids_per_fold[i]].copy()
        train_set_y = y_N[train_ids_per_fold[i]].copy()

        test_set_x = x_NF[test_ids_per_fold[i]].copy()
        test_set_y = y_N[test_ids_per_fold[i]].copy()

        estimator.fit(train_set_x, train_set_y)
        train_set_prediction = estimator.predict(train_set_x)
        test_set_prediction = estimator.predict(test_set_x)

        train_error = calc_root_mean_squared_error(
            train_set_y, train_set_prediction)
        test_error = calc_root_mean_squared_error(
            test_set_y, test_set_prediction)

        train_error_per_fold[i] = train_error
        test_error_per_fold[i] = test_error

    # for the provided estimator

    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    Examples
    --------
    >>> N = 11
    >>> n_folds = 3
    >>> tr_ids_per_fold, te_ids_per_fold = (
    ...     make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    >>> len(tr_ids_per_fold)
    3

    # Count of items in training sets
    >>> np.sort([len(tr) for tr in tr_ids_per_fold])
    array([7, 7, 8])

    # Count of items in the test sets
    >>> np.sort([len(te) for te in te_ids_per_fold])
    array([3, 4, 4])

    # Test ids should uniquely cover the interval [0, N)
    >>> np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    # Train ids should cover the interval [0, N) TWICE
    >>> np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
            8,  9,  9, 10, 10])
    '''
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state  # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # Obtain a shuffled order of the n_examples
    shuffled_ids = random_state.permutation(np.arange(n_examples))

    train_ids_per_fold = list()
    test_ids_per_fold = list()

    # Divide the labeled dataset into n-folds even-sized parts
    examples_per_fold = n_examples // n_folds
    remaining_examples = n_examples % n_folds  # Handle the remaining examples

    # Establish the row ids that belong to each fold's
    start_index = 0
    for i in range(n_folds):

        end_index = start_index + examples_per_fold + \
            (1 if i < remaining_examples else 0)

        test_fold = shuffled_ids[start_index: end_index].copy()

        train_fold = np.concatenate(
            (shuffled_ids[0: start_index], shuffled_ids[end_index:])).copy()

        start_index = end_index

        train_ids_per_fold.append(train_fold)
        test_ids_per_fold.append(test_fold)

    return train_ids_per_fold, test_ids_per_fold
