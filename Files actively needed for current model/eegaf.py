"""
=========================================================
EEGAF : EEG  Adaptation Framework Wrapper
=========================================================

Framework   : EEGAF
Version     : v1.0
Description : A backbone-agnostic wrapper that freezes 
              any pre-trained EEG foundation model 
              (e.g., LaBraM) and adapts its features 
              using lightweight downstream adapters.
=========================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from adapter import BaseAdapter, DomainAdapter, TopologyTemporalAdapter, AdapterConfig


class EEGAF(nn.Module):
    """
    EEG Foundation Adaptation Framework (EEGAF) Wrapper.
    
    This wrapper encapsulates a pre-trained EEG Foundation Model,
    manages the frozen backbone state, and exposes a trainable
    Domain Adapter combined with a task-specific classification head.
    """
    FRAMEWORK_VERSION = "EEGAF-v1.0"

    def __init__(
        self,
        foundation_model: nn.Module,
        num_classes: int,
        adapter: Optional[BaseAdapter] = None,
        adapter_config: Optional[AdapterConfig] = None,
    ):
        """
        Args:
            foundation_model: The pre-trained EEG foundation model (e.g., LaBraM).
            num_classes: Number of target classes for the downstream task.
            adapter: An instantiated EEGAF adapter. If None, a default DomainAdapter 
                     will be constructed using `adapter_config`.
            adapter_config: Configuration to instantiate the default DomainAdapter 
                            if no explicit adapter is provided.
        """
        super().__init__()
        self.foundation_model = foundation_model
        
        # 1. Retrieve embedding dimension dynamically from the foundation model using safe getattr fallback
        self.embed_dim = getattr(
            foundation_model, 
            "embed_dim", 
            getattr(foundation_model, "num_features", None)
        )
        if self.embed_dim is None:
            raise AttributeError(
                "Could not dynamically retrieve embedding dimension from foundation_model. "
                "Ensure it has an `embed_dim` or `num_features` attribute."
            )

        # 2. Setup the adapter (Ensure it matches embedding dimension)
        if adapter is not None:
            self.adapter = adapter
            if hasattr(adapter, "embed_dim") and adapter.embed_dim != self.embed_dim:
                raise ValueError(
                    f"Adapter embed_dim ({adapter.embed_dim}) does not match "
                    f"backbone embed_dim ({self.embed_dim})."
                )
        else:
            config = adapter_config if adapter_config is not None else AdapterConfig()

            if getattr(config, 'adapter_type', 'bottleneck') == 'topology_temporal':
                self.adapter = TopologyTemporalAdapter(
                    embed_dim=self.embed_dim,
                    config=config,
                    num_channels=config.num_channels,
                    graph_neighbors=config.graph_neighbors,
                )
            else:
                self.adapter = DomainAdapter(
                    embed_dim=self.embed_dim,
                    config=config
                )

        # 3. Explicitly construct and own the task-specific classification head
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.num_classes = num_classes
        
        # Initialize head weights
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # 4. Freeze backbone and explicitly guarantee that Adapter & Head remain trainable
        self.freeze_backbone()

    @property
    def backbone(self) -> nn.Module:
        """
        Syntactic sugar to quickly access the underlying foundation model.
        """
        return self.foundation_model

    def get_num_layers(self):
        return self.foundation_model.get_num_layers()

    def no_weight_decay(self):
        return self.foundation_model.no_weight_decay()

    def train(self, mode: bool = True) -> "EEGAF":
        """
        Overrides the standard PyTorch train behavior.
        Ensures the frozen foundation backbone stays strictly in evaluation mode
        to prevent dropout scaling issues and running stats drift in BatchNorm.
        """
        super().train(mode)
        # Force frozen backbone to always stay in evaluation mode during training
        self.foundation_model.eval()
        return self

    def freeze_backbone(self) -> None:
        """
        Freezes all parameters of the foundation model and explicitly
        enables training for the adapter and classification head.
        """
        for param in self.foundation_model.parameters():
            param.requires_grad = False
            
        # Guarantee adapter and head are always trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreezes the foundation model parameters to support full-model fine-tuning.
        Activates training state behaviors (e.g., Dropout/BatchNorm) for the backbone.
        """
        self.foundation_model.train()
        for param in self.foundation_model.parameters():
            param.requires_grad = True

    def replace_adapter(self, adapter: BaseAdapter) -> None:
        """
        Swaps out the current adapter with a new adapter module.
        Ensures proper trainability and training mode alignment.
        """
        if hasattr(adapter, "embed_dim") and adapter.embed_dim != self.embed_dim:
            raise ValueError(
                f"New adapter embed_dim ({adapter.embed_dim}) does not match "
                f"backbone embed_dim ({self.embed_dim})."
            )
        self.adapter = adapter
        
        # Match current training state of the parent framework
        self.adapter.train(self.training)
        
        # Ensure the new adapter parameters are trainable
        for param in self.adapter.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Extracts and adapts features from the input tensor, bypassing the classification head.
        Crucial for t-SNE, UMAP, and representation analysis.
        """
        # Call the instance directly to ensure PyTorch registers hooks and profiles correctly
        return self(x, return_features=True, **kwargs)

    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass through the EEGAF framework.
        
        Args:
            x: Input EEG tensor. Shape depends on the foundation model.
            return_features: If True, returns the adapted feature vector.
            kwargs: Extra keyword arguments for the backbone's `forward_features` function.
        """
        # Step 1: Extract features. Fail fast if the model isn't a proper foundation model.
        if not hasattr(self.foundation_model, "forward_features"):
            raise RuntimeError(
                "EEGAF requires the foundation model to implement 'forward_features()'. "
                "This ensures we isolate representation learning from classification heads."
            )
            
        if getattr(self.adapter, 'requires_patch_tokens', False):
            if kwargs.get('return_patch_tokens', False):
                raise ValueError('EEGAF reserves return_patch_tokens for topology-aware adapters.')
            features = self.foundation_model.forward_features(
                x, return_patch_tokens=True, **kwargs
            )
        else:
            features = self.foundation_model.forward_features(x, **kwargs)

        # Step 2: Adapt representations
        adapted_features = self.adapter(features)

        # If adapter returns a sequence of features (e.g., [batch, seq, dim]),
        # pool the sequence dimension to obtain a single vector per example.
        if adapted_features.ndim == 3:
            adapted_features = adapted_features.mean(dim=1)

        # Return adapted features directly when requested (useful for analysis)
        if return_features:
            return adapted_features

        # Step 4: Downstream Task Classification
        logits = self.head(adapted_features)
        return logits

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Returns a list of parameters currently set for optimization.
        """
        return [param for param in self.parameters() if param.requires_grad]

    def print_trainable_parameters(self):
        print("\n========== Trainable Parameters ==========\n")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name:60} {list(param.shape)}")

    def model_info(self) -> Dict[str, Any]:
        """
        Generates a summary of framework parameters for papers and experimental logs.
        Only counts trainable parameters for active modules to report accurate math.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.foundation_model.parameters())
        
        # Count parameters that are actually being optimized
        trainable_adapter_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        trainable_head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        return {
            "Framework": self.FRAMEWORK_VERSION,
            "Total Parameters": total_params,
            "Frozen Backbone Parameters": backbone_params,
            "Frozen Parameters (Total)": frozen_params,
            "Trainable Adapter Parameters": trainable_adapter_params,
            "Trainable Head Parameters": trainable_head_params,
            "Total Trainable Parameters": trainable_params,
            "PEFT Tuning Ratio (%)": (trainable_params / total_params) * 100,
            "Adapter Info": self.adapter.adapter_info() if hasattr(self.adapter, "adapter_info") else "Unknown"
        }
