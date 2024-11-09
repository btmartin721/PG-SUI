import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """Class for a custom dataset that includes a boolean mask for missing values.

    This class is a custom dataset that includes a boolean mask for missing values. The mask is used to validate the predictions of imputation models by comparing only the values at the locations where missing values occur.

    Example:
        >>> dataset = CustomTensorDataset(X_tensor, y_tensor, missing_mask_tensor)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for X_batch, y_batch, mask_batch in dataloader:
        >>>     predictions = model(X_batch)
        >>>     loss = criterion(predictions[mask_batch], y_batch[mask_batch])

    Attributes:
        X_tensor (torch.Tensor): Input tensor.
        y_tensor (torch.Tensor): Target tensor (ground truth values for missing data).
        missing_mask_tensor (torch.Tensor): Boolean mask with True where missing data occurs.
    """

    def __init__(self, X_tensor, y_tensor, missing_mask_tensor=None):
        """A custom dataset that includes a boolean mask for missing values.

        This class is a custom dataset that includes a boolean mask for missing values. The mask is used to validate the predictions of imputation models by comparing only the values at the locations where missing values occur.

        Args:
            X_tensor (torch.Tensor or array-like): Input tensor.
            y_tensor (torch.Tensor or array-like): Target tensor (ground truth values for missing data).
            missing_mask_tensor (torch.Tensor or array-like, optional): Boolean mask with True where missing data occurs.
        """
        self.X_tensor = self._to_tensor(X_tensor)
        self.y_tensor = self._to_tensor(y_tensor)
        self.missing_mask_tensor = (
            self._to_tensor(missing_mask_tensor)
            if missing_mask_tensor is not None
            else None
        )

        # Ensure missing_mask_tensor has boolean values if provided.
        if self.missing_mask_tensor is not None:
            self.missing_mask_tensor = self.missing_mask_tensor.bool()

    def _to_tensor(self, data):
        """Convert data to torch.Tensor if it isn't already."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data)
        return data

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X_tensor)

    def __getitem__(self, idx):
        """Retrieve a sample and its target from the dataset.

        This method retrieves a sample and its target from the dataset. If a missing mask is provided, it also returns the mask.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input sample, target, and mask (if available).
        """
        X_sample = self.X_tensor[idx]
        y_sample = self.y_tensor[idx]
        if self.missing_mask_tensor is not None:
            mask = self.missing_mask_tensor[idx]
            return X_sample, y_sample, mask
        return X_sample, y_sample
