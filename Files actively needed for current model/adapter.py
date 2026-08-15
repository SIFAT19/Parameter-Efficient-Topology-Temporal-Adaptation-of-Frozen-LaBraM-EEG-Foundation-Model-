"""
=========================================================
EEGAF- :  Adapter
=========================================================

Framework : EEGAF (EEG Adaptation Framework)

Description
-----------

This adapter is backbone-independent and can be reused
for future EEG foundation models.

Future Versions
---------------

=========================================================
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    """Configuration object for DomainAdapter."""

    reduction_ratio: int = 6
    dropout: float = 0.1
    init_scale: float = 1e-3
    adapter_type: str = 'bottleneck'
    num_channels: int = 32
    graph_neighbors: int = 4


class BaseAdapter(nn.Module):
    """
    Base class for all EEGAF adapters.

    Future adapters should inherit from this class.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Every adapter must implement the forward() function."
        )

    def adapter_info(self) -> dict:
        """
        Returns basic adapter information.
        """

        trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

        return {
            "Adapter": self.__class__.__name__,
            "Trainable Parameters": trainable
        }


class DomainAdapter(BaseAdapter):
    """
    EEGAF-v1 Domain Adapter

    Architecture
    ------------
        Input
          │
          ▼
      LayerNorm
          │
          ▼
      Linear Down
          │
          ▼
         GELU
          │
          ▼
       Dropout
          │
          ▼
      Linear Up
          │
          ▼
     Learnable Scale (α)
          │
          ▼
      Residual Add
          │
          ▼
        Output
    """

    def __init__(
        self,
        embed_dim: int,
        config: AdapterConfig | None = None,
    ):

        super().__init__()

        if config is None:
            config = AdapterConfig()
        elif not isinstance(config, AdapterConfig):
            raise TypeError("config must be an AdapterConfig instance.")

        reduction_ratio = config.reduction_ratio
        dropout = config.dropout
        init_scale = config.init_scale

        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive.")

        hidden_dim = max(embed_dim // reduction_ratio, 8)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.reduction_ratio = reduction_ratio
        self.config = config

        # -----------------------------------------
        # Layers
        # -----------------------------------------

        self.norm = nn.LayerNorm(embed_dim)

        self.down_proj = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.up_proj = nn.Linear(
            hidden_dim,
            embed_dim
        )

        # Learnable residual scaling
        self.alpha = nn.Parameter(
            torch.ones(1) * init_scale
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for Linear layers.
        """

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor

            Shape:
                [Batch, Embed_Dim]
                or
                [Batch, Tokens, Embed_Dim]

        Returns
        -------
        Tensor

            Same shape as input.
        """

        residual = x

        out = self.norm(x)

        out = self.down_proj(out)

        out = self.activation(out)

        out = self.dropout(out)

        out = self.up_proj(out)

        out = self.alpha * out

        out = residual + out

        return out

    def extra_repr(self):

        return (
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"reduction_ratio={self.reduction_ratio}, "
            f"alpha={self.alpha.item():.6f}"
        )

    def adapter_info(self) -> dict:
        """
        Extended adapter information.
        """

        info = super().adapter_info()

        info.update({
            "Embedding Dimension": self.embed_dim,
            "Hidden Dimension": self.hidden_dim,
            "Reduction Ratio": self.reduction_ratio,
            "Dropout": self.dropout.p,
            "Alpha": float(self.alpha.detach().cpu())
        })

        return info


class TopologyTemporalAdapter(BaseAdapter):
    """A parameter-efficient adapter for EEG channel-patch representations.

    The frozen foundation model provides one token for every EEG channel and
    temporal patch.  This adapter retains that structure instead of adapting
    only a globally pooled vector: a small depth-wise temporal convolution
    models local patch dynamics and a fixed montage graph exchanges information
    among neighbouring electrodes.  Only this module and the task head are
    optimized.

    The canonical DEAP 32-channel 10-20 montage is used by default.  The
    adapter deliberately validates token layout at runtime so it cannot silently
    apply an incorrect channel graph to another montage.
    """

    # Approximate 2-D locations of DEAP's 32 EEG electrodes.  They are used
    # only to define a fixed nearest-neighbour graph; no test-subject data are
    # used to construct the graph.
    DEAP_10_20_COORDS = (
        (-0.50, 1.00), (-0.30, 0.80), (-0.30, 0.50), (-0.85, 0.50),
        (-0.80, 0.20), (-0.30, 0.20), (-0.30, 0.00), (-1.00, 0.00),
        (-0.80, -0.20), (-0.30, -0.20), (-0.30, -0.50), (-0.85, -0.50),
        (-0.30, -0.80), (-0.50, -1.00), (0.00, -1.00), (0.00, -0.50),
        (0.50, 1.00), (0.30, 0.80), (0.30, 0.50), (0.85, 0.50),
        (0.80, 0.20), (0.30, 0.20), (0.00, 0.00), (0.30, 0.00),
        (0.30, -0.20), (0.80, -0.20), (0.30, -0.50), (0.85, -0.50),
        (0.30, -0.80), (0.50, -1.00), (0.00, 0.80), (0.00, 0.50),
    )

    def __init__(
        self,
        embed_dim: int,
        config: AdapterConfig | None = None,
        num_channels: int = 32,
        graph_neighbors: int = 4,
    ):
        super().__init__()
        if config is None:
            config = AdapterConfig()
        if num_channels != 32:
            raise ValueError(
                'TopologyTemporalAdapter currently supports the canonical '
                '32-channel DEAP montage only.'
            )
        if not 1 <= graph_neighbors < num_channels:
            raise ValueError('graph_neighbors must be between 1 and num_channels - 1.')

        self.embed_dim = embed_dim
        self.hidden_dim = max(embed_dim // config.reduction_ratio, 8)
        self.num_channels = num_channels
        self.graph_neighbors = graph_neighbors
        self.requires_patch_tokens = True

        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, self.hidden_dim)
        # Depth-wise convolution keeps the adapter small while modelling the
        # ordered temporal patches within each EEG channel.
        self.temporal_conv = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1,
            groups=self.hidden_dim, bias=False,
        )
        self.spatial_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.up_proj = nn.Linear(self.hidden_dim, embed_dim)
        self.alpha = nn.Parameter(torch.ones(1) * config.init_scale)
        self.temporal_alpha = nn.Parameter(torch.ones(1) * config.init_scale)
        self.spatial_alpha = nn.Parameter(torch.ones(1) * config.init_scale)
        self.register_buffer('adjacency', self._make_adjacency(), persistent=True)
        self._init_weights()

    def _make_adjacency(self) -> torch.Tensor:
        coords = torch.tensor(self.DEAP_10_20_COORDS, dtype=torch.float32)
        distances = torch.cdist(coords, coords)
        distances.fill_diagonal_(float('inf'))
        neighbours = distances.topk(self.graph_neighbors, largest=False).indices
        adjacency = torch.zeros(self.num_channels, self.num_channels)
        adjacency.scatter_(1, neighbours, 1.0)
        # Symmetrize and row-normalize for stable graph aggregation.
        adjacency = torch.maximum(adjacency, adjacency.T)
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                'TopologyTemporalAdapter expects [batch, channel_patches, embed_dim] tokens.'
            )
        batch_size, token_count, embed_dim = x.shape
        if embed_dim != self.embed_dim or token_count % self.num_channels != 0:
            raise ValueError(
                f'Expected [batch, {self.num_channels} * time_patches, {self.embed_dim}], '
                f'got {tuple(x.shape)}.'
            )
        time_patches = token_count // self.num_channels
        residual = x
        hidden = self.down_proj(self.norm(x)).view(
            batch_size, self.num_channels, time_patches, self.hidden_dim
        )

        temporal = self.temporal_conv(
            hidden.permute(0, 1, 3, 2).reshape(
                batch_size * self.num_channels, self.hidden_dim, time_patches
            )
        ).reshape(batch_size, self.num_channels, self.hidden_dim, time_patches).permute(0, 1, 3, 2)
        hidden = hidden + self.temporal_alpha * temporal

        # Exchange each channel's summary only with fixed neighbouring DEAP
        # electrodes, then broadcast the spatial correction to its time patches.
        channel_summary = hidden.mean(dim=2)
        neighbours = torch.einsum('ij,bjh->bih', self.adjacency, channel_summary)
        hidden = hidden + self.spatial_alpha * self.spatial_proj(neighbours).unsqueeze(2)

        delta = self.up_proj(self.dropout(self.activation(hidden))).reshape(
            batch_size, token_count, embed_dim
        )
        return residual + self.alpha * delta

    def adapter_info(self) -> dict:
        info = super().adapter_info()
        info.update({
            'Adapter': self.__class__.__name__,
            'Embedding Dimension': self.embed_dim,
            'Hidden Dimension': self.hidden_dim,
            'Channels': self.num_channels,
            'Graph Neighbours': self.graph_neighbors,
        })
        return info
