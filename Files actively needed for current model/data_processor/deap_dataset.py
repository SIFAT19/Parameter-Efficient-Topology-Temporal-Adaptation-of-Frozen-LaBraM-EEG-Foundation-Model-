"""
=========================================================
EEGAF : DEAP Dataset Loader (Scalable & Normalized)
=========================================================

Framework   : EEGAF
Version     : v1.2

Description
-----------
Memory-efficient PyTorch dataset for the DEAP emotion recognition dataset.
Uses an index-based lazy loading strategy with per-window z-score normalization.

Features
--------
- Zero eager-loading of windows (O(1) memory footprint during init)
- On-demand window slicing and channel-wise z-score normalization
- On-demand subject caching to balance memory and disk I/O
- Leave-One-Subject-Out (LOSO) validation support
- Strict parameter bounds checking

=========================================================
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly

from .base_dataset import BaseEEGDataset


class DEAPDataset(BaseEEGDataset):
    """
    EEGAF DEAP Dataset (Lazy-Loaded & Normalized)

    Parameters
    ----------
    root_dir : str
        Directory containing s01.dat ... s32.dat

    train : bool
        True -> training subjects
        False -> testing subject

    leave_out_subject : int
        Subject used for testing (1-32)

    task : str
        One of: "valence", "arousal", "dominance", "liking"

    threshold : float
        Threshold used for binary classification.

    window_size : int
        Sliding window length.

    stride : int
        Sliding window stride.

    use_only_eeg : bool
        Keep only first 32 EEG channels.
    """

    # DEAP trials contain a 3-second pre-trial baseline at 128 Hz.  It is not
    # part of the affective-response period and must not be used as a sample.
    BASELINE_SAMPLES = 3 * 128

    CHANNEL_NAMES = [
        "Fp1", "AF3", "F3", "F7",
        "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7",
        "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4",
        "F8", "FC6", "FC2", "Cz",
        "C4", "T8", "CP6", "CP2",
        "P4", "P8", "PO4", "O2"
    ]

    LABEL_INDEX = {
        "valence": 0,
        "arousal": 1,
        "dominance": 2,
        "liking": 3
    }

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        leave_out_subject: int = 1,
        task: str = "valence",
        threshold: float = 5.0,
        window_size: int = 512,
        stride: int = 256,
        use_only_eeg: bool = True,
    ):
        super().__init__(
            window_size=window_size,
            stride=stride,
        )

        # 4. Validate leave_out_subject
        if not (1 <= leave_out_subject <= 32):
            raise ValueError(
                "leave_out_subject must be between 1 and 32."
            )

        self.root_dir = Path(root_dir)
        self.train = train
        self.leave_out_subject = leave_out_subject
        self.task = task.lower()
        self.threshold = threshold
        self.use_only_eeg = use_only_eeg

        if self.task not in self.LABEL_INDEX:
            raise ValueError(f"Unknown task: {task}")

        # Metadata index store: (subject_file_path, trial_idx, start_sample, label)
        self.window_metadata: List[Tuple[Path, int, int, int]] = []

        # Memory cache to avoid reading from disk constantly during training
        self._subject_cache: Dict[Path, dict] = {}

        self._build_index()

    def _build_index(self):
        """
        Builds a lightweight index of metadata for every window across
        all valid subjects.
        """
        subject_files = sorted(self.root_dir.glob("s*.dat"))

        if len(subject_files) == 0:
            raise RuntimeError(f"No DEAP files found in {self.root_dir}")

        for subject_file in subject_files:
            subject_id = int(subject_file.stem[1:])

            # Apply LOSO protocol filtering
            if self.train and subject_id == self.leave_out_subject:
                continue
            if not self.train and subject_id != self.leave_out_subject:
                continue

            self._index_subject(subject_file)

    def _index_subject(self, subject_path: Path):
        """
        Briefly reads the file structure to calculate window offsets and 
        pre-computes trial labels.
        """
        with open(subject_path, "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        data_shape = subject["data"].shape  # (trials, channels, samples)
        labels = subject["labels"]          # (trials, 4)

        num_trials = data_shape[0]
        total_samples = data_shape[2]

        # 5. Verify window size against trial length
        if self.window_size > total_samples:
            raise ValueError(
                f"window_size ({self.window_size}) is larger than the EEG trial length ({total_samples})."
            )

        for trial in range(num_trials):
            label = self._convert_label(labels[trial])

            # Calculate sliding window start indices
            for start in range(
                self.BASELINE_SAMPLES,
                total_samples - self.window_size + 1,
                self.stride
            ):
                self.window_metadata.append(
                    (subject_path, trial, start, label)
                )

    def _get_subject_data(self, subject_path: Path) -> dict:
        """
        Helper method to get subject data. Uses a cache so we don't hit 
        the disk on every single __getitem__ call.
        """
        if subject_path not in self._subject_cache:
            with open(subject_path, "rb") as f:
                subject = pickle.load(f, encoding="latin1")
            
            # Keep only EEG channels immediately to save cache space
            if self.use_only_eeg:
                subject["data"] = subject["data"][:, :32, :]
                
            self._subject_cache[subject_path] = subject

        return self._subject_cache[subject_path]

    def _convert_label(self, label_vector) -> int:
        """
        Convert DEAP score into a binary label.
        """
        score = label_vector[self.LABEL_INDEX[self.task]]
        return 1 if score >= self.threshold else 0

    def __len__(self) -> int:
        """
        Return total number of EEG windows.
        """
        return len(self.window_metadata)

    # 3. Add return type hint
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Lazily loads, slices, and normalizes the requested window.
        """
        # Retrieve metadata for the requested index
        subject_path, trial_idx, start_sample, label_val = self.window_metadata[index]

        # Fetch subject data (from cache if already loaded once)
        subject_data = self._get_subject_data(subject_path)
        
        # Slice the requested window on-the-fly
        end_sample = start_sample + self.window_size
        window_data = subject_data["data"][trial_idx, :, start_sample:end_sample]

        # Resample DEAP windows from 128 Hz to 200 Hz: 32 x 512 -> 32 x 800.
        window_data = resample_poly(window_data, up=25, down=16, axis=1)

        # 2. Normalize each EEG window (Per-channel z-score normalization)
        window_data = (window_data - window_data.mean(axis=1, keepdims=True)) / (
            window_data.std(axis=1, keepdims=True) + 1e-8
        )
        window_data = window_data.reshape(32, 4, 200)

        # Convert to PyTorch Tensors
        sample = torch.tensor(window_data, dtype=torch.float32)
        label = torch.tensor(label_val, dtype=torch.long)

        return (
            sample,
            label,
            self.CHANNEL_NAMES
        )

    def get_channel_names(self) -> List[str]:
        """
        Return DEAP EEG channel names.
        """
        return self.CHANNEL_NAMES

    @property
    def num_channels(self) -> int:
        return len(self.CHANNEL_NAMES)

    @property
    def sampling_rate(self) -> int:
        return 200

    @property
    def num_classes(self) -> int:
        return 1

    # 6. Add dataset_name
    @property
    def dataset_name(self) -> str:
        return "DEAP"
