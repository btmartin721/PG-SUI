from pgsui.impute.unsupervised.base import BaseNNImputer


class SAE(BaseNNImputer):
    def __init__(
        self,
        **kwargs,
    ):
        self.num_classes = 4
        self.is_multiclass_ = True if self.num_classes != 4 else False
        self.activate = None
        self.nn_method_ = "SAE"
        self.act_func_ = None
        self.testing = kwargs.get("testing", False)

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
        )

    def run_sae(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run standard autoencoder using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes (training dataset) with known and missing values of shape (n_samples, n_features).

            y_train (numpy.ndarray): Onehot-encoded genotypes (training dataset) with known and missing values of shape (n_samples, n_features, num_classes.)

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()
        scoring = None

        histories = list()
        models = list()

        (
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )


class UBP(BaseNNImputer):
    def __init__(
        self,
        *,
        nlpca=False,
        **kwargs,
    ):
        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.nlpca = nlpca
        self.nn_method_ = "NLPCA" if self.nlpca else "UBP"
        self.num_classes = 4
        self.is_multiclass_ = True if self.num_classes != 4 else False
        self.testing = kwargs.get("testing", False)
        self.activate = None
        self.act_func_ = None

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
            nlpca=self.nlpca,
        )

    def run_nlpca(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        y_train = model_params.pop("y_train")
        ubp_weights = None
        phase = None

        (
            V,
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            ubp_weights=ubp_weights,
            phase=phase,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def run_ubp(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run UBP using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        search_n_components = False

        y_train = model_params.pop("y_train")

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_,
                self.sim_missing_mask_,
                num_classes=self.num_classes,
            )

            if "n_components" in self.gridparams:
                search_n_components = True
                n_components_searched = self.n_components
        else:
            scoring = None

        for phase in range(1, 4):
            ubp_weights = models[1].get_weights() if phase == 3 else None

            (
                V,
                model,
                best_history,
                best_params,
                best_score,
                best_clf,
                search,
                metrics,
            ) = self.run_clf(
                y_train,
                y_true,
                model_params,
                compile_params,
                fit_params,
                ubp_weights=ubp_weights,
                phase=phase,
                testing=False,
            )

            if phase == 1:
                # Cannot have V input with different n_components
                # in other phases than are in phase 1.
                # So the n_components search has to happen in phase 1.
                if best_params is not None and search_n_components:
                    n_components_searched = best_params["n_components"]
                    model_params["V"] = {n_components_searched: model.V_latent.copy()}
                    model_params["n_components"] = n_components_searched
                    self.n_components = n_components_searched
                    self.gridparams.pop("n_components")

                else:
                    model_params["V"] = V
            elif phase == 2:
                model_params["V"] = V

            elif phase == 3:
                if best_params is not None and search_n_components:
                    best_params["n_components"] = n_components_searched

            histories.append(best_history)
            models.append(model)
            del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def _initV(self, y_train, search_mode):
        """Initialize random input V as dictionary of numpy arrays.

        Args:
            y_train (numpy.ndarray): One-hot encoded training dataset (actual data).

            search_mode (bool): Whether doing grid search.

        Returns:
            Dict[int, numpy.ndarray]: Dictionary with n_components: V as key-value pairs.

        Raises:
            ValueError: Number of components must be >= 2.
        """
        vinput = dict()
        if search_mode:
            if "n_components" in self.gridparams:
                n_components = self.gridparams["n_components"]
            else:
                n_components = self.n_components

            if not isinstance(n_components, int):
                if min(n_components) < 2:
                    raise ValueError(
                        f"n_components must be >= 2, but a value of {n_components} was specified."
                    )

                elif len(n_components) == 1:
                    vinput[n_components[0]] = self.nn_.init_weights(
                        y_train.shape[0], n_components[0]
                    )

                else:
                    for c in n_components:
                        vinput[c] = self.nn_.init_weights(y_train.shape[0], c)
            else:
                vinput[self.n_components] = self.nn_.init_weights(
                    y_train.shape[0], self.n_components
                )

        else:
            vinput[self.n_components] = self.nn_.init_weights(
                y_train.shape[0], self.n_components
            )

        return vinput
