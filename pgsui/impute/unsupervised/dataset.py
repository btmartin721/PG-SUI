from typing import List, Tuple

import torch
from snpio.utils.logging import LoggerManager
from torch.utils.data import TensorDataset


class CustomTensorDataset(TensorDataset):
    """A custom dataset extending the functionality of a PyTorch TensorDataset.

    This dataset class ensures that the input data are torch.Tensor objects and sets attributes for data and target. It also provides methods to retrieve the shape of the data and target tensors, as well as to split the dataset into train, validation, and test sets.

    Example:
        >>> dataset = CustomTensorDataset(data, target)
        >>> train_dataset, val_dataset, test_dataset = dataset.split_dataset(train_ratio=0.7, val_ratio=0.1, seed=42)
        >>> print(dataset.get_data_shape())
        >>> print(dataset.get_target_shape())
        >>> print(len(dataset))

    Attributes:
        data (torch.Tensor): The input data tensor.
        target (torch.Tensor): The target tensor.
    """

    def __init__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
        *,
        logger: LoggerManager = None,
        indices: List[int] = None,
    ):
        """Initializes the CustomTensorDataset with data and target tensors.

        This dataset class ensures that the input data are torch.Tensor objects and sets attributes for data and target. It also provides methods to retrieve the shape of the data and target tensors, as well as to split the dataset into train, validation, and test sets.

        Args:
            data (torch.Tensor): The input data tensor.
            target (torch.Tensor): The target tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
            logger (LoggerManager, optional): Logger for logging messages. Defaults to None.
            indices (List[int], optional): List of indices to use for the dataset. Defaults to None.

        Raises:
            TypeError: If the input data or target are not torch.Tensor objects.
        """
        if logger is not None:
            self.logger = logger
        else:
            logman = LoggerManager(
                name=__name__, prefix="pgsui", debug=False, verbose=0
            )
            self.logger = logman.get_logger()

        # Check if inputs are torch tensors
        if not isinstance(data, torch.Tensor):
            msg = f"Expected data to be torch.Tensor, got {type(data)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if not isinstance(target, torch.Tensor):
            msg = f"Expected target to be torch.Tensor, got {type(target)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if not isinstance(mask, torch.Tensor):
            msg = f"Expected mask to be torch.Tensor, got {type(mask)}"
            self.logger.error(msg)
            raise TypeError

        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)

        # Set attributes for data and target
        self.data = data
        self.target = target
        self.mask = mask
        self._indices = indices

        # Set the tensors attribute for easy access
        self._tensors = (self.data, self.target)

        # Initialize the parent class (TensorDataset)
        super().__init__(self.data, self.target)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Returns a single sample of data and target from the dataset."""
        data_sample = self.data[idx]
        target_sample = self.target[idx]
        mask_sample = self.mask[idx]

        # Debugging log
        if data_sample.shape[0] != self.data.shape[1]:
            self.logger.warning(
                f"Data sample {idx} has incorrect shape: {data_sample.shape}"
            )

        return data_sample, target_sample, mask_sample

    def get_data_shape(self):
        """Returns the shape of the data tensor.

        Returns:
            torch.Size: Shape of the data tensor.
        """
        return self.data.shape

    def get_target_shape(self):
        """Returns the shape of the target tensor.

        Returns:
            torch.Size: Shape of the target tensor.
        """
        return self.target.shape

    def get_mask_shape(self):
        """Returns the shape of the mask tensor.

        Returns:
            torch.Size: Shape of the mask tensor.
        """
        return self.mask.shape

    @property
    def indices(self) -> List[int]:
        """Returns the indices of the subset dataset."""
        return self._indices

    @indices.setter
    def indices(self, indices: List[int]):
        """Sets the indices of the subset dataset."""
        self._indices = indices

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the data and target tensors."""
        return self._tensors

    def tensors(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        """Sets the data, target, and mask tensors."""
        self.data, self.target = tensors
        self._tensors = tensors
        self._indices = None
        super().__init__(self.data, self.target)
        self.logger.info("Updated data and target tensors.")
