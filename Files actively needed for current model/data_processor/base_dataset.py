"""
=========================================================
EEGAF : Base EEG Dataset
=========================================================

Framework   : EEGAF
Version     : v1.0

Description
-----------
Abstract base dataset for all EEG datasets.

Every EEG dataset (DEAP, SEED, DREAMER, AMIGOS, TUAB, etc.)
should inherit from this class.

=========================================================
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class BaseEEGDataset(Dataset, ABC):
    """
    Base class for all EEG datasets used in EEGAF.
    """

    def __init__(
        self,
        window_size: int,
        stride: int,
    ):
        super().__init__()

        self.window_size = window_size
        self.stride = stride

    @abstractmethod
    def __len__(self) -> int:
        """
        Return number of EEG samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        """
        Must return

        sample,
        label,
        input_chans
        """
        pass

    @abstractmethod
    def get_channel_names(self) -> List[str]:
        """
        Return EEG channel names.
        """
        pass

    @property
    @abstractmethod
    def num_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass