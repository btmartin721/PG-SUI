import sys

import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    hamming_loss,
    make_scorer,
    precision_recall_curve,
    average_precision_score,
    multilabel_confusion_matrix,
    f1_score,
)

from sklearn.preprocessing import label_binarize

try:
    from ..impute.unsupervised.neural_network_methods import (
        NeuralNetworkMethods,
    )
except (ModuleNotFoundError, ValueError, ImportError):
    from impute.unsupervised.neural_network_methods import NeuralNetworkMethods


class Scorers:
    @staticmethod
    def compute_roc_auc_micro_macro(
        y_true, y_pred, num_classes=3, binarize_pred=True
    ):
        """Compute ROC curve with AUC scores.

        ROC (Receiver Operating Characteristic) curves and AUC (area under curve) scores are computed per-class and for micro and macro averages.

        Args:
            y_true (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,). y_true should be integer-encoded.

            y_pred (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,). y_pred should be probabilities.

            num_classes (int, optional): How many classes to use. Defaults to 3.

            binarize_pred (bool, optional): Whether to binarize y_pred. If False, y_pred should be probabilities of each class. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary with true and false positive rates along probability threshold curve per class, micro and macro tpr and fpr curves averaged across classes, and AUC scores per-class and for micro and macro averages.
        """
        cats = range(num_classes)

        # Get only classes that appear in y_true.
        classes = [i for i in cats if i in y_true]

        # Binarize the output for use with ROC-AUC.
        y_true_bin = label_binarize(y_true, classes=cats)

        if binarize_pred:
            y_pred_bin = label_binarize(y_pred, classes=cats)
        else:
            y_pred_bin = y_pred

        for i in range(y_true_bin.shape[1]):
            if i not in classes:
                y_true_bin = np.delete(y_true_bin, i, axis=-1)
                y_pred_bin = np.delete(y_pred_bin, i, axis=-1)

        n_classes = len(classes)

        # Compute ROC curve and ROC area for each class.
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, c in enumerate(classes):
            fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[c] = auc(fpr[c], tpr[c])

        # Compute micro-average ROC curve and ROC area.
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred_bin.ravel()
        )

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))

        # Then interpolate all ROC curves at these points.
        mean_tpr = np.zeros_like(all_fpr)
        for i in classes:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally, average it and compute AUC.
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr

        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_auc["fpr_macro"] = fpr["macro"]
        roc_auc["tpr_macro"] = tpr["macro"]
        roc_auc["fpr_micro"] = fpr["micro"]
        roc_auc["tpr_micro"] = tpr["micro"]

        for i in classes:
            roc_auc[f"fpr_{i}"] = fpr[i]
            roc_auc[f"tpr_{i}"] = tpr[i]

        return roc_auc

    @staticmethod
    def compute_pr(y_true, y_pred, use_int_encodings=False, num_classes=3):
        """Compute precision-recall curve with Average Precision scores.

        PR and AP scores are computed per-class and for micro and macro averages.

        Args:
            y_true (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,).

            y_pred (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,). y_pred should be integer-encoded.

            use_int_encodings (bool, optional): Whether the imputer model is a neural network model. Defaults to False.

            num_classes (int, optional): How many classes to use. Defaults to 3.

         Returns:
            Dict[str, Any]: Dictionary with precision and recall curves per class and micro and macro averaged across classes, plus AP scores per-class and for micro and macro averages.
        """

        cats = range(num_classes)

        is_multiclass = True if num_classes != 4 else False

        # Get only classes that appear in y_true.
        classes = [i for i in cats if i in y_true]

        # Binarize the output for use with ROC-AUC.
        y_true_bin = label_binarize(y_true, classes=cats)
        y_pred_proba_bin = y_pred

        if is_multiclass:
            for i in range(y_true_bin.shape[1]):
                if i not in classes:
                    y_true_bin = np.delete(y_true_bin, i, axis=-1)
                    y_pred_proba_bin = np.delete(y_pred_proba_bin, i, axis=-1)

        nn = NeuralNetworkMethods()
        if use_int_encodings:
            y_pred_012 = nn.decode_masked(y_true_bin, y_pred_proba_bin)
            thresh = 0.5
        else:
            y_pred_012 = nn.decode_masked(
                y_true_bin,
                y_pred_proba_bin,
                is_multiclass=is_multiclass,
                return_int=False,
                return_multilab=True,
            )

            encode_func = (
                nn.encode_multiclass if is_multiclass else nn.encode_multilab
            )

            y_true = encode_func(y_true, num_classes=num_classes)

        # Make confusion matrix to get true negatives and true positives.
        mcm = multilabel_confusion_matrix(y_true, y_pred_012)

        tn = np.sum(mcm[:, 0, 0])
        tn /= num_classes

        tp = np.sum(mcm[:, 1, 1])
        tp /= num_classes

        baseline = tp / (tn + tp)

        precision = dict()
        recall = dict()
        average_precision = dict()

        for i, c in enumerate(classes):
            precision[c], recall[c], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba_bin[:, i]
            )
            average_precision[c] = average_precision_score(
                y_true_bin[:, i], y_pred_proba_bin[:, i]
            )

        # A "micro-average": quantifying score on all classes jointly.
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_pred_proba_bin.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_true_bin, y_pred_proba_bin, average="micro"
        )

        average_precision["macro"] = average_precision_score(
            y_true_bin, y_pred_proba_bin, average="macro"
        )

        if use_int_encodings:
            y_pred_012 = (
                nn.decode_masked(
                    y_true_bin,
                    y_pred_proba_bin,
                    threshold=thresh,
                    return_multilab=True,
                    predict_still_missing=False,
                ),
            )

        f1 = f1_score(y_true_bin, y_pred_012, average="macro")

        # Aggregate all recalls
        all_recall = np.unique(np.concatenate([recall[i] for i in classes]))

        # Then interpolate all PR curves at these points.
        mean_precision = np.zeros_like(all_recall)
        for i in classes:
            mean_precision += np.interp(all_recall, precision[i], recall[i])

        # Finally, average it and compute AUC.
        mean_precision /= num_classes

        recall["macro"] = all_recall
        precision["macro"] = mean_precision

        results = dict()

        results["micro"] = average_precision["micro"]
        results["macro"] = average_precision["macro"]
        results["f1_score"] = f1
        results["recall_macro"] = all_recall
        results["precision_macro"] = mean_precision
        results["recall_micro"] = recall["micro"]
        results["precision_micro"] = precision["micro"]

        for i in classes:
            results[f"recall_{i}"] = recall[i]
            results[f"precision_{i}"] = precision[i]
            results[i] = average_precision[i]
        results["baseline"] = baseline

        return results

    @staticmethod
    def check_if_tuple(y_pred):
        """Checks if y_pred is a tuple and if so, returns the first element of the tuple."""
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        return y_pred

    @staticmethod
    def accuracy_scorer(y_true, y_pred, **kwargs):
        """Get accuracy score for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities. They must first be decoded to use with accuracy_score.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).

        missing_mask = kwargs.get("missing_mask")
        testing = kwargs.get("testing", False)
        nn_model = kwargs.get("nn_model", True)

        y_pred = Scorers.check_if_tuple(y_pred)

        if nn_model:
            nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if nn_model:
            y_pred_masked_decoded = nn.decode_masked(
                y_true_masked, y_pred_masked, predict_still_missing=False
            )
        else:
            y_pred_masked_decoded = y_pred_masked

        acc = accuracy_score(y_true_masked, y_pred_masked_decoded)

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)
            print(y_pred_masked_decoded)

        return acc

    @staticmethod
    def hamming_scorer(y_true, y_pred, **kwargs):
        """Get Hamming score for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities. They must first be decoded to use with hamming_scorer.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).

        missing_mask = kwargs.get("missing_mask")
        testing = kwargs.get("testing", False)
        nn_model = kwargs.get("nn_model", True)
        num_classes = kwargs.get("num_classes", 3)

        y_pred = Scorers.check_if_tuple(y_pred)

        if nn_model:
            nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if nn_model:
            y_pred_masked_decoded = nn.decode_masked(
                y_true_masked,
                y_pred_masked,
                predict_still_missing=False,
            )
        else:
            y_pred_masked_decoded = y_pred_masked

        ham = hamming_loss(y_true_masked, y_pred_masked_decoded)

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)
            print(y_pred_masked_decoded)

        return ham

    @staticmethod
    def auc_macro(y_true, y_pred, **kwargs):
        """Get AUC score with macro averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        num_classes = kwargs.get("num_classes", 3)
        nn_model = kwargs.get("nn_model", True)
        testing = kwargs.get("testing", False)

        is_multiclass = True if num_classes != 4 else False

        y_pred = Scorers.check_if_tuple(y_pred)

        if nn_model:
            nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if nn_model:
            y_pred_masked_decoded = nn.decode_masked(
                y_true_masked, y_pred_masked, is_multiclass=is_multiclass
            )
        else:
            y_pred_masked_decoded = y_pred_masked

        roc_auc = Scorers.compute_roc_auc_micro_macro(
            y_true_masked,
            y_pred_masked,
            num_classes=num_classes,
            binarize_pred=False,
        )

        return roc_auc["macro"]

    @staticmethod
    def auc_micro(y_true, y_pred, **kwargs):
        """Get AUC score with micro averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        nn_model = kwargs.get("nn_model", True)
        num_classes = kwargs.get("num_classes", 3)

        is_multiclass = True if num_classes != 4 else False

        y_pred = Scorers.check_if_tuple(y_pred)

        if nn_model:
            nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if nn_model:
            y_pred_masked_decoded = nn.decode_masked(
                y_true_masked, y_pred_masked, is_multiclass=is_multiclass
            )
        else:
            y_pred_masked_decoded = y_pred_masked

        roc_auc = Scorers.compute_roc_auc_micro_macro(
            y_true_masked,
            y_pred_masked,
            num_classes=num_classes,
            binarize_pred=False,
        )

        return roc_auc["micro"]

    @staticmethod
    def pr_macro(y_true, y_pred, **kwargs):
        """Get Precision-Recall score with macro averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """

        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        y_pred = Scorers.check_if_tuple(y_pred)

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        pr_ap = Scorers.compute_pr(
            y_true_masked, y_pred_masked, num_classes=num_classes
        )

        return pr_ap["macro"]

    @staticmethod
    def pr_samples(y_true, y_pred, **kwargs):
        """Get Precision-Recall score with samples averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        y_pred = Scorers.check_if_tuple(y_pred)

        nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        pr_ap = Scorers.compute_pr(
            y_true_masked, y_pred_masked, num_classes=num_classes
        )

        return pr_ap["samples"]

    @staticmethod
    def f1_samples(y_true, y_pred, **kwargs):
        """Get F1 score with samples averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        num_classes = kwargs.get("num_classes", 3)

        y_pred = Scorers.check_if_tuple(y_pred)

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        pr_ap = Scorers.compute_pr(
            y_true_masked, y_pred_masked, num_classes=num_classes
        )

        return pr_ap["f1_score"]

    @staticmethod
    def pr_micro(y_true, y_pred, **kwargs):
        """Get Precision-Recall score with micro averaging for grid search.

        If provided, only calculates score where missing_mask is True (i.e., data were missing). This is so that users can simulate missing data for known values, and then the predictions for only those known values can be evaluated.

        Args:
            y_true (numpy.ndarray): 012-encoded true target values.

            y_pred (tensorflow.EagerTensor): Predictions from model as probabilities.

            kwargs (Any): Keyword arguments to use with scorer. Supported options include ``missing_mask`` and ``testing``\.

        Returns:
            float: Metric score by comparing y_true and y_pred.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        y_pred = Scorers.check_if_tuple(y_pred)

        nn = NeuralNetworkMethods()

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        pr_ap = Scorers.compute_pr(
            y_true_masked, y_pred_masked, num_classes=num_classes
        )

        return pr_ap["micro"]

    @classmethod
    def make_multimetric_scorer(
        cls, metrics, missing_mask, num_classes=3, testing=False
    ):
        """Get all scoring metrics and make an sklearn scorer.

        Args:
            metrics (str or List[str]): Metrics to use with grid search. If string, it will be converted to a list of one element.

            missing_mask (numpy.ndarray): Missing mask to use to demarcate values to use with scoring.

            num_classes (int, optional): How many classes to use. Defaults to 3.

            testing (bool, optional): True if in test mode, wherein it prints y_true and y_pred_decoded as 1D lists for comparison. Otherwise False. Defaults to False.
        Returns:
            Dict[str, Callable]: Dictionary with callable scoring functions to use with grid search as the values.

        Raises:
            ValueError: Invalid scoring metric provided.
        """
        if isinstance(metrics, str):
            metrics = [metrics]

        scorers = dict()
        for item in metrics:
            if item.lower() == "accuracy":
                scorers["accuracy"] = make_scorer(
                    cls.accuracy_scorer,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "hamming":
                scorers["hamming"] = make_scorer(
                    cls.hamming_scorer,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                    greater_is_better=False,
                )
            elif item.lower() == "auc_macro":
                scorers["auc_macro"] = make_scorer(
                    cls.auc_macro,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "auc_micro":
                scorers["auc_micro"] = make_scorer(
                    cls.auc_micro,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "precision_recall_macro":
                scorers["precision_recall_macro"] = make_scorer(
                    cls.pr_macro,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "precision_recall_micro":
                scorers["precision_recall_micro"] = make_scorer(
                    cls.pr_micro,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "precision_recall_samples":
                scorers["precision_recall_samples"] = make_scorer(
                    cls.pr_samples,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            elif item.lower() == "f1_score":
                scorers["f1_score"] = make_scorer(
                    cls.f1_samples,
                    missing_mask=missing_mask,
                    num_classes=num_classes,
                    testing=testing,
                )
            else:
                raise ValueError(f"Invalid scoring_metric provided: {item}")
        return scorers

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get("missing_mask")
        nn_model = kwargs.get("nn_model", True)
        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        is_multiclass = True if num_classes != 4 else False

        if nn_model:
            nn = NeuralNetworkMethods()

        # VAE has tuple output.
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        roc_auc = Scorers.compute_roc_auc_micro_macro(
            y_true_masked,
            y_pred_masked,
            num_classes=num_classes,
            binarize_pred=False,
        )

        pr_ap = Scorers.compute_pr(
            y_true_masked,
            y_pred_masked,
            num_classes=num_classes,
        )

        acc = accuracy_score(
            y_true_masked,
            nn.decode_masked(
                y_true_masked,
                y_pred_masked,
                is_multiclass=is_multiclass,
                return_int=True,
            ),
        )
        ham = hamming_loss(
            y_true_masked,
            nn.decode_masked(
                y_true_masked,
                y_pred_masked,
                is_multiclass=is_multiclass,
                return_int=True,
            ),
        )

        if testing:
            y_pred_masked_decoded = nn.decode_masked(
                y_true_masked,
                y_pred_masked,
                is_multiclass=is_multiclass,
                return_int=True,
            )

            bin_mapping = [np.array2string(x) for x in y_pred_masked]

            with open("genotype_dist.csv", "w") as fout:
                fout.write(
                    "site,prob_vector,imputed_genotype,expected_genotype\n"
                )
                for i, (yt, yp, ypd) in enumerate(
                    zip(y_true_masked, bin_mapping, y_pred_masked_decoded)
                ):
                    fout.write(f"{i},{yp},{ypd},{yt}\n")
            # np.set_printoptions(threshold=np.inf)
            # print(y_true_masked)
            # print(y_pred_masked_decoded)

        metrics = dict()
        metrics["accuracy"] = acc
        metrics["roc_auc"] = roc_auc
        metrics["precision_recall"] = pr_ap
        metrics["hamming"] = ham

        return metrics
