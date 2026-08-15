# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import copy
import datetime
import numpy as np
import random
import time
import torch
import torch.backends.cudnn as cudnn
import json

import os
import shutil

from pathlib import Path
from collections import OrderedDict, Counter
from torch.utils.data import WeightedRandomSampler
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune
from eegaf import EEGAF
from adapter import AdapterConfig
from metrics import (
    calculate_metrics,
    save_metrics,
    save_confusion_matrix,
    save_predictions,
    append_results_csv,
)

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='G:\Reseacch2\LaBraM-main\output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--results_file', default='results/results.csv',
                        help='CSV file used for aggregate held-out-subject metrics.')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of DataLoader workers.')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    
    # Issue 1: Set default dataset to "DEAP"
    parser.add_argument('--dataset', default='DEAP', type=str,
                        help='dataset: DEAP | TUAB | TUEV')

    # DEAP dataset arguments
    parser.add_argument('--deap_root', type=str, default='./DEAP')
    parser.add_argument('--task', type=str, default='valence',
                        choices=['valence', 'arousal', 'dominance', 'liking'])
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--leave_out_subject', type=int, default=1)
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of DEAP training trials reserved for validation.')
    parser.add_argument('--selection_metric', type=str, default='balanced_accuracy',
                        choices=['balanced_accuracy', 'roc_auc', 'f1', 'accuracy'],
                        help='Validation metric used to select the checkpoint. The test set is evaluated once.')
    parser.add_argument('--overfit_sanity_steps', type=int, default=0,
                        help='If > 0, verify that a cloned model can overfit one training batch before the main run.')

    # EEGAF Adapter parameters
    parser.add_argument(
        "--adapter_reduction",
        type=int,
        default=16,
        help="Reduction ratio for EEGAF adapter"
    )

    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.1,
        help="Dropout inside EEGAF adapter"
    )

    parser.add_argument(
        "--adapter_alpha",
        type=float,
        default=0.1,
        help="Initial residual scaling factor"
    )
    parser.add_argument(
        "--adapter_type",
        choices=['bottleneck', 'topology_temporal'],
        default='bottleneck',
        help="Adapter architecture. topology_temporal is designed for DEAP's 32-channel montage."
    )
    parser.add_argument(
        "--graph_neighbors",
        type=int,
        default=4,
        help="Nearest DEAP electrodes used by the topology_temporal adapter."
    )

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except ImportError:
            raise ImportError(
                "DeepSpeed is not installed. "
                "Install it only if you use --enable_deepspeed."
            )
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def deap_collate_fn(batch):
    """
    Custom collate for DEAPDataset.
    Each item is (Tensor, Tensor, List[str] of 32 channel names).
    All items share the same channel layout, so ch_names must NOT
    be batched/transposed by default_collate — just take one copy.
    """
    samples, labels, ch_names_list = zip(*batch)
    samples = torch.stack(samples, dim=0)
    labels = torch.stack(labels, dim=0)
    ch_names = ch_names_list[0]
    return samples, labels, ch_names


def split_deap_by_trial(dataset, validation_fraction, seed):
    """Split DEAP data by whole trials, never by overlapping EEG windows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError('--val_fraction must be between 0 and 1.')

    trial_windows = {}
    for index, (subject_path, trial_idx, _start, label) in enumerate(dataset.window_metadata):
        key = (str(subject_path), trial_idx)
        trial_windows.setdefault(label, {}).setdefault(key, []).append(index)

    rng = random.Random(seed)
    validation_indices = []
    training_indices = []
    for label, trials in trial_windows.items():
        groups = list(trials.values())
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * validation_fraction))
        # Preserve at least one trial per class for training.
        n_val = min(n_val, len(groups) - 1) if len(groups) > 1 else 0
        for group in groups[:n_val]:
            validation_indices.extend(group)
        for group in groups[n_val:]:
            training_indices.extend(group)

    if not training_indices or not validation_indices:
        raise RuntimeError('Trial-level validation split is empty; check the DEAP dataset and val_fraction.')
    return (torch.utils.data.Subset(dataset, training_indices),
            torch.utils.data.Subset(dataset, validation_indices))


def run_overfit_sanity_check(model, batch, device, steps):
    """Diagnostic only: prove the adapter/head can fit one fixed mini-batch."""
    if steps <= 0:
        return
    samples, targets, batch_ch_names = batch
    samples = samples.float().to(device)
    targets = targets.float().unsqueeze(-1).to(device)
    input_chans = utils.get_input_chans(batch_ch_names)
    probe = copy.deepcopy(model).to(device)
    probe.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in probe.parameters() if parameter.requires_grad], lr=1e-3
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    initial_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(samples, input_chans=input_chans), targets)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(probe(samples, input_chans=input_chans))
        accuracy = ((probabilities >= 0.5) == targets.bool()).float().mean().item()
    print(
        f'Overfit sanity check ({steps} steps): loss {initial_loss:.4f} -> {loss.item():.4f}, '
        f'accuracy {accuracy:.4f}'
    )


def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("path/to/TUEV")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == "DEAP":
        from data_processor.deap_dataset import DEAPDataset
        # Load the full training set and create a held-out validation split
        full_train_dataset = DEAPDataset(
            root_dir=args.deap_root,
            train=True,
            leave_out_subject=args.leave_out_subject,
            task=args.task,
            threshold=args.threshold,
        )
        test_dataset = DEAPDataset(
            root_dir=args.deap_root,
            train=False,
            leave_out_subject=args.leave_out_subject,
            task=args.task,
            threshold=args.threshold,
        )
        # Keep whole trials together. Random window splitting would leak 50%-overlapping
        # neighbouring windows from the same trial into train and validation.
        train_dataset, val_dataset = split_deap_by_trial(
            full_train_dataset, args.val_fraction, args.seed
        )
        ch_names = full_train_dataset.get_channel_names()
        args.nb_classes = full_train_dataset.num_classes
        metrics = ["accuracy", "balanced_accuracy", "f1_weighted"]
        
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        print('Set torch multiprocessing sharing strategy to file_system')
    except Exception as exc:
        print(f'Could not set multiprocessing sharing strategy: {exc}')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Issue 4: Restored GPU optimization setting
    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    sample_weights = None
    if args.nb_classes == 1:
        train_labels = None
        if isinstance(dataset_train, torch.utils.data.Subset):
            base_dataset = dataset_train.dataset
            if hasattr(base_dataset, 'window_metadata'):
                train_labels = [base_dataset.window_metadata[i][3] for i in dataset_train.indices]
        elif hasattr(dataset_train, 'window_metadata'):
            train_labels = [w[3] for w in dataset_train.window_metadata]
        elif hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'window_metadata'):
            train_labels = [w[3] for w in dataset_train.dataset.window_metadata]

        if train_labels is not None and len(train_labels) > 0:
            train_labels = [int(label) for label in train_labels]
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[int(label)] for label in train_labels]
            print(f"Computed class-balanced sample weights: {class_weights.tolist()}")
        else:
            print("Warning: could not compute class-balanced sampler weights; using regular sampler.")

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if utils.is_main_process():
        def _dataset_label_counts(dataset, name):
            if dataset is None:
                return f"{name}: None"
            try:
                labels = [int(y.item() if hasattr(y, 'item') else y) for _, y, *_ in dataset]
                return f"{name}: {Counter(labels)}"
            except Exception as exc:
                return f"{name}: failed to inspect labels ({exc})"

        print(_dataset_label_counts(dataset_train, 'Train'))
        print(_dataset_label_counts(dataset_val, 'Val'))
        print(_dataset_label_counts(dataset_test, 'Test'))

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if sample_weights is not None and not args.distributed:
            sampler_train = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print("Sampler_train = WeightedRandomSampler(class-balanced)")
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if dataset_val is not None and len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_val is not None else None
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_test is not None else None
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Added persistent_workers dynamically (True only if num_workers > 0)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=deap_collate_fn,
    )

    if dataset_val is not None:
        eval_num_workers = args.num_workers if os.name != 'nt' else 0
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=eval_num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=(eval_num_workers > 0),
            collate_fn=deap_collate_fn,
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=eval_num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=(eval_num_workers > 0),
                collate_fn=deap_collate_fn,
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=eval_num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=(eval_num_workers > 0),
                collate_fn=deap_collate_fn,
            )
    else:
        data_loader_val = None
        data_loader_test = None

    foundation_model = get_models(args)

    patch_size = foundation_model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(
            args.finetune,
             map_location="cpu",
             weights_only=False
    )

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        # LaBraM pretraining stores its final LayerNorm as ``norm``.  This
        # fine-tuning model uses mean pooling, for which ``norm`` is an
        # Identity and the active output LayerNorm is ``fc_norm``.  Without
        # this compatibility mapping the pretrained final normalization is
        # silently unused and fc_norm stays at its fresh initialization.  The
        # foundation model is frozen by EEGAF, so it cannot later correct this.
        if getattr(foundation_model, 'fc_norm', None) is not None:
            for suffix in ('weight', 'bias'):
                source_key = f'norm.{suffix}'
                target_key = f'fc_norm.{suffix}'
                if source_key in checkpoint_model:
                    if checkpoint_model[source_key].shape != foundation_model.state_dict()[target_key].shape:
                        raise RuntimeError(
                            f'Cannot map {source_key} to {target_key}: '
                            f'{tuple(checkpoint_model[source_key].shape)} != '
                            f'{tuple(foundation_model.state_dict()[target_key].shape)}'
                        )
                    checkpoint_model[target_key] = checkpoint_model.pop(source_key)
                    print(f'Mapped pretrained {source_key} to {target_key}')

        state_dict = foundation_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(foundation_model, checkpoint_model, prefix=args.model_prefix)

    adapter_cfg = AdapterConfig(
        reduction_ratio=args.adapter_reduction,
        dropout=args.adapter_dropout,
        init_scale=args.adapter_alpha,
        adapter_type=args.adapter_type,
        num_channels=len(ch_names),
        graph_neighbors=args.graph_neighbors,
    )

    model = EEGAF(
        foundation_model=foundation_model,
        num_classes=args.nb_classes,
        adapter_config=adapter_cfg,
    )

    model.to(device)

    if args.overfit_sanity_steps:
        if args.nb_classes != 1:
            raise ValueError('--overfit_sanity_steps currently supports the binary DEAP setup only.')
        run_overfit_sanity_check(
            model, next(iter(data_loader_train)), device, args.overfit_sanity_steps
        )

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print(model.model_info())
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        print("Trainable Parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel()}")
                
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        # Build optimizer using only trainable parameters exposed via a small proxy.
        class _TrainableProxy(object):
            def __init__(self, module):
                self._module = module

            def named_parameters(self):
                for name, p in self._module.named_parameters():
                    if p.requires_grad:
                        yield name, p

            def parameters(self):
                for p in self._module.parameters():
                    if p.requires_grad:
                        yield p

            def no_weight_decay(self):
                return self._module.no_weight_decay() if hasattr(self._module, 'no_weight_decay') else set()

        proxy_model = _TrainableProxy(model_without_ddp)

        optimizer = create_optimizer(
            args, proxy_model, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    if args.eval:
        balanced_accuracy = []
        accuracy = []
        loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]
        for data_loader in loaders:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    if args.output_dir and utils.is_main_process():
        config = vars(args)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    start_time = time.time()
    best_selection_score = float('-inf')
    best_epoch = None
    best_model_state = None
    initial_trainable_params = {
        name: parameter.detach().clone()
        for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad
    }

    def trainable_parameter_change(prefix):
        squared_change = 0.0
        for name, parameter in model_without_ddp.named_parameters():
            if name.startswith(prefix) and name in initial_trainable_params:
                squared_change += torch.sum(
                    (parameter.detach() - initial_trainable_params[name]).float().pow(2)
                ).item()
        return squared_change ** 0.5
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )
        
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy'] * 100:.2f}%")
            selection_score = float(val_stats[args.selection_metric])
            if not np.isfinite(selection_score):
                selection_score = float('-inf')
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_epoch = epoch
                # Model selection uses validation data only; the held-out subject
                # is evaluated once after the training loop.
                best_model_state = copy.deepcopy(model_without_ddp.state_dict())
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                    checkpoint_src = os.path.join(args.output_dir, "checkpoint-best.pth")
                    checkpoint_dst = os.path.join(args.output_dir, "best_model.pth")
                    if os.path.exists(checkpoint_src):
                        shutil.copy(checkpoint_src, checkpoint_dst)
                print(f"New best validation {args.selection_metric}: {best_selection_score:.4f} at epoch {epoch + 1}")

            learning_diagnostics = {
                'adapter_parameter_change_l2': trainable_parameter_change('adapter.'),
                'head_parameter_change_l2': trainable_parameter_change('head.'),
                'backbone_trainable_parameters': sum(
                    p.numel() for p in model_without_ddp.foundation_model.parameters() if p.requires_grad
                ),
            }
            print('Learning diagnostics:', learning_diagnostics)
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key in ['accuracy', 'balanced_accuracy', 'f1_weighted', 'pr_auc', 'roc_auc', 'cohen_kappa', 'loss']:
                        log_writer.update(**{key: value}, head="val", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                          **{f'val_{k}': v for k, v in val_stats.items() if k not in ['y_true', 'y_pred', 'y_prob']},
                          **learning_diagnostics,
                          'selection_metric': args.selection_metric,
                          'selection_score': selection_score,
                          'epoch': epoch,
                          'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if best_model_state is None:
        raise RuntimeError('No valid validation checkpoint was selected; cannot evaluate the test set.')

    # Restore the validation-selected model and evaluate the LOSO test subject once.
    model_without_ddp.load_state_dict(best_model_state)
    primary_test_loader = data_loader_test[0] if isinstance(data_loader_test, list) else data_loader_test
    test_stats = evaluate(
        primary_test_loader, model, device, header='Final test:', ch_names=ch_names,
        metrics=metrics, is_binary=args.nb_classes == 1,
    )
    metric_results, cm = calculate_metrics(
        test_stats['y_true'], test_stats['y_pred'], test_stats['y_prob'],
        num_classes=args.nb_classes if args.nb_classes > 1 else 2,
    )
    save_metrics(metric_results, args.output_dir)
    save_confusion_matrix(cm, args.output_dir)
    save_predictions(test_stats['y_true'], test_stats['y_pred'], test_stats['y_prob'], args.output_dir)
    append_results_csv(
        metric_results,
        task=args.task,
        subject=args.leave_out_subject,
        epoch=best_epoch + 1,
        output_file=args.results_file,
    )
    print(f"Final held-out test at validation-selected epoch {best_epoch + 1}: {metric_results}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
