from .metahmm import *
import copy



class MetaLearningTask(object):

    def __init__(self, cfg: Optional[MetaLearningConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(MetaLearningConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg
        self.cfg_train = copy.deepcopy(cfg)
        self.cfg_val = copy.deepcopy(cfg)


        self.cfg_train.data.seed = self.cfg_val.data.seed*49152

        self.data_train = MetaHMM(self.cfg_train.data)
        self.data_train.to_device("cpu")

        self.data_val = MetaHMM(self.cfg_val.data)
        self.data_val.to_device("cpu")

        self.seen_tokens = torch.tensor(0).to("cuda")


    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = self.data_train
        self.val_data = self.data_val

    def _collate_and_prepare(self, batch):
        """Custom collate function that returns prepared batches from prepare_step()

        This function is used as the collate_fn in the dataloaders to automatically
        apply prepare_step() transformations to each batch.

        Args:
            batch: List of batch dicts from __getitems__ (each with batch_size=1)
                   OR a single batch dict (if DataLoader directly uses __getitems__)

        Returns:
            tuple: (shifted_idx, shifted_labels, envs_torch, states) from prepare_step()
        """
        # Handle case where batch is a list of dicts (from Subset wrapper)
        if isinstance(batch, list):
            # Each element is a dict with batch_size=1, need to concatenate
            if len(batch) == 1:
                batch_dict = batch[0]
            else:
                # Concatenate multiple single-sample batches
                batch_dict = {}
                for key in batch[0].keys():
                    if isinstance(batch[0][key], torch.Tensor):
                        batch_dict[key] = torch.cat([b[key] for b in batch], dim=0)
                    elif isinstance(batch[0][key], (jnp.ndarray, np.ndarray)):
                        # Convert JAX/numpy arrays to torch for concatenation
                        tensors = [torch.from_numpy(np.array(b[key])) for b in batch]
                        batch_dict[key] = torch.cat(tensors, dim=0)
                    else:
                        # For other types, try to stack
                        batch_dict[key] = torch.stack([b[key] for b in batch])
        else:
            # Batch is already a dict
            batch_dict = batch

        return self.prepare_step(batch_dict)

    def train_dataloader(self, use_prepare_step=True, drop_last=True):
        """Create training dataloader

        Args:
            use_prepare_step: bool, if True uses _collate_and_prepare to return
                            preprocessed batches, otherwise returns raw batch dicts
            drop_last: bool, if True drops the last incomplete batch (important for sharding)
        """
        collate_fn = self._collate_and_prepare if use_prepare_step else lambda x: x
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    def val_dataloader(self, use_prepare_step=True, drop_last=True):
        """Create validation dataloader

        Args:
            use_prepare_step: bool, if True uses _collate_and_prepare to return
                            preprocessed batches, otherwise returns raw batch dicts
            drop_last: bool, if True drops the last incomplete batch (important for sharding)
        """
        collate_fn = self._collate_and_prepare if use_prepare_step else lambda x: x
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=drop_last,
        )

    def prepare_step(self, batch):
        """
        Prepare batch for training by shifting tokens and handling masks

        Returns:
            tuple: (shifted_idx, shifted_labels, envs_torch, states, intv_envs_torch, intv_idx_torch)
        """

        bs = batch["input_ids"].shape[0]

        # Shift tokens, labels and mask - ensure contiguous for JAX compatibility
        shifted_idx = batch["input_ids"][..., :-1].contiguous()
        shifted_labels = batch["input_ids"][..., 1:].contiguous()

        # Apply pad mask
        if "ignore_mask" in batch.keys():
            shifted_labels[batch["ignore_mask"][..., 1:]] = IGNORE_INDEX
            # Ensure still contiguous after in-place operation
            shifted_labels = shifted_labels.contiguous()

        # Count the number of non-padding tokens seen
        self.seen_tokens += torch.sum(shifted_labels != IGNORE_INDEX)

        # Make states contiguous for JAX compatibility
        states = batch["states"][..., :-1].contiguous()

        # envs are indices corresponding to the HMM ids
        # Check if already torch tensor to avoid unnecessary conversion
        if isinstance(batch["envs"], torch.Tensor):
            envs_torch = batch["envs"].contiguous()
        else:
            envs_torch = j2t(batch["envs"])

        # Handle intervention data if present
        if "intv_envs" in batch:
            if isinstance(batch["intv_envs"], torch.Tensor):
                intv_envs_torch = batch["intv_envs"].contiguous()
                intv_idx_torch = batch["intv_idx"].contiguous() if batch["intv_idx"] is not None else None
            else:
                intv_envs_torch = j2t(batch["intv_envs"])
                intv_idx_torch = None if batch["intv_idx"] is None else j2t(batch["intv_idx"])
        else:
            # No intervention data present
            intv_envs_torch = None
            intv_idx_torch = None

        return shifted_idx, shifted_labels, envs_torch, states, intv_envs_torch, intv_idx_torch


# ============================================================================
# Wrapper Functions for dataset_wrappers.py compatibility
# ============================================================================

def metalearning_hmm(batch_size,
                     use_prepare_step=True,
                     drop_last=True,
                     n_states=30,
                     n_obs=60,
                     context_length=(200, 200),
                     context_length_dist="uniform",
                     adjust_varlen_batch=False,
                     start_at_n=None,
                     seed=42,
                     base_cycles=4,
                     base_directions=2,
                     base_speeds=3,
                     cycle_families=4,
                     group_per_family=2,
                     cycle_per_group=3,
                     family_directions=2,
                     family_speeds=2,
                     emission_groups=4,
                     emission_group_size=2,
                     emission_shifts=2,
                     emission_edge_per_node=3,
                     emission_noise=1e-5,
                     dt=None,
                     root=None,
                     prng_key=None,
                     **dl_kwargs):
    '''
    Meta-learning Hidden Markov Model dataset with DataLoader wrapper from Gagnon et al.

    This function wraps the MetaLearningTask class which provides proper PyTorch DataLoaders
    with train/val splits for the MetaHMM task.

    **Arguments:**
    - batch_size: int, the batch size for the dataloaders
    - use_prepare_step: bool, if True returns preprocessed batches (shifted_idx, shifted_labels, envs, states),
                       if False returns raw batch dicts (default: True)
    - drop_last: bool, if True drops the last incomplete batch to ensure all batches have the same size.
                Important when using data parallelism/sharding (default: True)
    - n_states: int, number of hidden states in the HMM
    - n_obs: int, number of possible observations
    - context_length: tuple of int, (min, max) sequence length
    - context_length_dist: str, distribution for sampling sequence lengths ("uniform")
    - adjust_varlen_batch: bool, whether to adjust variable length batches
    - start_at_n: int or None, starting position for training
    - seed: int, random seed for dataset generation
    - base_cycles: int, number of base cycles in transition structure
    - base_directions: int, number of directions for base cycles
    - base_speeds: int, number of speeds for base cycles
    - cycle_families: int, number of cycle families
    - group_per_family: int, groups per cycle family
    - cycle_per_group: int, cycles per group
    - family_directions: int, directions for family cycles
    - family_speeds: int, speeds for family cycles
    - emission_groups: int, number of emission groups
    - emission_group_size: int, size of each emission group
    - emission_shifts: int, number of emission shifts
    - emission_edge_per_node: int, edges per node in emission graph
    - emission_noise: float, noise level for emissions
    - dt: float, not used for this dataset
    - root: str, not used for this dataset
    - prng_key: int, random seed (alternative to seed parameter)
    - dl_kwargs: dict, additional keyword arguments (not used for MetaLearningTask)

    **Returns:**
    - dataloader_train: PyTorch DataLoader for training
    - dataloader_test: PyTorch DataLoader for testing (same as validation)
    - dataloader_val: PyTorch DataLoader for validation
    - input_size: int, vocabulary size (n_obs)
    - output_size: int, vocabulary size (n_obs)

    **Usage:**
    When use_prepare_step=True (default):
        shifted_idx, shifted_labels, envs, states = next(iter(dataloader_train))

    When use_prepare_step=False:
        batch_dict = next(iter(dataloader_train))[0]
        # batch_dict contains: 'input_ids', 'states', 'envs', and optionally 'ignore_mask'
    '''
    if prng_key is not None:
        seed = prng_key

    # Create MetaHMM configuration
    metahmm_config = MetaHMMConfig(
        n_states=n_states,
        n_obs=n_obs,
        context_length=context_length,
        context_length_dist=context_length_dist,
        adjust_varlen_batch=adjust_varlen_batch,
        start_at_n=start_at_n,
        seed=seed,
        base_cycles=base_cycles,
        base_directions=base_directions,
        base_speeds=base_speeds,
        cycle_families=cycle_families,
        group_per_family=group_per_family,
        cycle_per_group=cycle_per_group,
        family_directions=family_directions,
        family_speeds=family_speeds,
        emission_groups=emission_groups,
        emission_group_size=emission_group_size,
        emission_shifts=emission_shifts,
        emission_edge_per_node=emission_edge_per_node,
        emission_noise=emission_noise,
    )

    # Create MetaLearning configuration
    metalearning_config = MetaLearningConfig(
        data=metahmm_config,
        batch_size=batch_size,
        val_size=None, #Not used
        val_ratio=None, #Not used
    )

    # Create task and setup dataloaders
    task = MetaLearningTask(metalearning_config)
    task.setup()

    # Get the dataloaders with optional prepare_step and drop_last
    dataloader_train = task.train_dataloader(use_prepare_step=use_prepare_step, drop_last=drop_last)
    dataloader_val = task.val_dataloader(use_prepare_step=use_prepare_step, drop_last=drop_last)
    dataloader_test = dataloader_val  # Use validation for test as well

    # Return dataloaders and input/output sizes
    return dataloader_train, dataloader_test, dataloader_val, n_obs, n_obs

if __name__ == '__main__':
    print("=" * 80)
    print("Example 1: Standard MetaHMM")
    print("=" * 80)

    # Create dataloaders using wrapper function
    train_dl, test_dl, val_dl, input_size, output_size = metalearning_hmm(
        batch_size=256,
        val_size=1000,
        n_states=6,
        n_obs=10,
        cycle_families=4,
        context_length=(200, 200),
        seed=42
    )

    print(f"Input size: {input_size}, Output size: {output_size}")
    batch = next(iter(train_dl))
    print(f"Batch is a tuple with {len(batch)} elements")
    print(f"  shifted_idx shape: {batch[0].shape}")
    print(f"  shifted_labels shape: {batch[1].shape}")
    print(f"  envs_torch shape: {batch[2].shape}")
    print(f"  states shape: {batch[3].shape}")

    print("\n" + "=" * 80)
    print("Example 2: FSC MetaHMM with interventions")
    print("=" * 80)


