from typing import List, Tuple

import torch
from snpio.utils.logging import LoggerManager
from torch.utils.data import Subset, TensorDataset, random_split


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
        *,
        logger: LoggerManager = None,
        indices: List[int] = None,
    ):
        """Initializes the CustomTensorDataset with data and target tensors.

        This dataset class ensures that the input data are torch.Tensor objects and sets attributes for data and target. It also provides methods to retrieve the shape of the data and target tensors, as well as to split the dataset into train, validation, and test sets.

        Args:
            data (torch.Tensor): The input data tensor.
            target (torch.Tensor): The target tensor.
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

        # Set attributes for data and target
        self.data = data
        self.target = target
        self._indices = indices

        # Initialize the parent class (TensorDataset)
        super().__init__(self.data, self.target)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Returns a single sample of data and target from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and target tensors at the given index.
        """
        return self.data[idx], self.target[idx]

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

    def split_dataset(
        self, train_ratio: float, val_ratio: float, seed: int = None
    ) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """Splits the dataset into train, validation, and test datasets.

        This method splits the dataset into train, validation, and test datasets based on the provided ratios. Half of the validation set is used for testing. The seed parameter can be set to ensure reproducibility of the split. The method returns the train, validation, and test datasets as TensorDataset objects.

        Args:
            train_ratio (float): Proportion of the dataset to use for the training set. Must be between 0.0 and 1.0.
            val_ratio (float): Proportion of the dataset to use for the validation set. This value is halved to form the validation and test sets equally. Must be between 0.0 and 1.0.
            seed (int, optional): Seed for random shuffling. Defaults to None.

        Returns:
            Tuple[TensorDataset, TensorDataset, TensorDataset]: Train, validation, and test datasets.

        Raises:
            ValueError: If the sum of train_ratio and val_ratio does not equal 1.0.
        """
        if train_ratio + val_ratio != 1.0:
            msg = f"Train and validation ratios must sum to 1.0, but got: {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Set the seed if provided, to ensure reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Calculate the lengths of the train, validation, and test sets
        total_size = len(self.data)
        train_size = int(total_size * train_ratio)
        remaining_size = total_size - train_size

        # Calculate the sizes for validation and test sets (equal halves from
        # remaining)
        test_ratio = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
        val_size = int(remaining_size * test_ratio // 2)
        test_size = remaining_size - val_size

        # Split the dataset into train, validation, and test sets
        sizes = [train_size, val_size, test_size]
        train_dataset, val_dataset, test_dataset = random_split(self, sizes)

        self.logger.info(
            f"Split dataset into train: {len(train_dataset)}, validation: {len(val_dataset)}, test: {len(test_dataset)} sets."
        )

        return train_dataset, val_dataset, test_dataset

    def get_data_from_subset(self, subset: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the data and target tensors from a subset by using its indices.

        Args:
            subset (Subset): A subset of the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and target tensors corresponding to the subset.
        """
        indices = subset.indices if self.indices is None else self.indices
        data_subset = self.data[indices]
        target_subset = self.target[indices]
        return data_subset, target_subset

    @property
    def indices(self) -> List[int]:
        """Returns the indices of the subset dataset."""
        return self._indices

    @indices.setter
    def indices(self, indices: List[int]):
        """Sets the indices of the subset dataset."""
        self._indices = indices
