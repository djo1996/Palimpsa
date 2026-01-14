#!/usr/bin/env python3
"""
Standalone MetaHMM and FSC MetaHMM dataset implementation.
Combined from src/dataset_wrappers.py and src/tasks/gagnon_etalhmm.py

This file provides:
- MetaHMM: Hidden Markov Model dataset for meta-learning
- FSCLearningTask: Few-Shot Continual learning variant with interventions
- Wrapper functions compatible with training scripts

Original authors: Leo Gagnon, Emre Neftci
Modified: Combined into standalone file
"""

import math
import multiprocessing as mp
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, singledispatchmethod
from itertools import product
from math import gcd
from multiprocessing.connection import Connection
from typing import *

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamax.hidden_markov_model import CategoricalHMM
from dynamax.hidden_markov_model.models.categorical_hmm import (
    ParamsCategoricalHMM,
    ParamsCategoricalHMMEmissions,
    ParamsStandardHMMInitialState,
    ParamsStandardHMMTransitions,
)
from dynamax.hidden_markov_model.parallel_inference import FilterMessage, HMMPosteriorFiltered, _condition_on, lax
from jax.scipy.special import logsumexp
from numpy.random._generator import Generator
from omegaconf import MISSING, OmegaConf
from scipy.special import softmax
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, StackDataset, Subset, TensorDataset
from tqdm import tqdm

# Simple conversion functions to replace torch2jax
def j2t(jax_array):
    """Convert JAX array to PyTorch tensor"""
    return torch.from_numpy(np.array(jax_array))

def t2j(torch_tensor):
    """Convert PyTorch tensor to JAX array"""
    return jnp.array(torch_tensor.detach().cpu().numpy())

IGNORE_INDEX = -100
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class MetaHMMConfig:
    tag: Optional[str] = None
    n_states: int = 30
    n_obs: int = 60
    context_length: Tuple[int] = (200, 200)
    context_length_dist: str = "uniform"
    adjust_varlen_batch: bool = False
    start_at_n: Optional[int] = None
    seed: int = 42
    base_cycles: int = 4
    base_directions: int = 2
    base_speeds: int = 3
    cycle_families: int = 4
    group_per_family: int = 2
    cycle_per_group: int = 3
    family_directions: int = 2
    family_speeds: int = 2
    emission_groups: int = 4
    emission_group_size: int = 2
    emission_shifts: int = 2
    emission_edge_per_node: int = 3
    emission_noise: float = 1e-5


@dataclass
class MetaLearningConfig:
    data: MetaHMMConfig
    batch_size: int
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    lr: Optional[float] = 1e-3
    intv_idx: Optional[Union[int, List[int], Tuple[int, int]]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def cycle_to_transmat(cycle: List[int], n_states: int) -> np.array:
    transition = np.zeros(shape=(n_states, n_states), dtype=np.int16)
    for i in range(len(cycle)):
        transition[
            cycle[i],
            cycle[(i + 1) % (len(cycle))],
        ] = 1.0
    return transition


def preferential_attachement_edges(
    states: np.array, obs: np.array, n: int, generator: Generator
) -> List[Tuple[int]]:

    states_ids = np.arange(len(states))
    obs_ids = np.arange(len(obs))
    obs_degree = np.zeros_like(obs_ids)

    # Add initial edges from each state to a random obs
    init_obs = generator.choice(obs_ids, size=len(states), replace=False)
    obs_degree[init_obs] = 1
    edges = [(states[i], obs[init_obs[i]]) for i in range(len(states))]

    # Iteratively add <n> edges, sampling state at random and obs at random weighted by degree
    for i in range(n):
        s_id = generator.choice(states_ids)
        o_id = generator.choice(obs_ids, p=softmax(obs_degree))

        obs_degree[o_id] += 1
        edges.append((states[s_id], obs[o_id]))

    return edges


# ============================================================================
# MetaHMM Dataset
# ============================================================================

class MetaHMM(Dataset):
    def __init__(self, cfg: MetaHMMConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.generator = np.random.default_rng(cfg.seed)
        self.index_to_latent = jnp.array(
            self._make_index_to_latent(), device=jax.devices("cpu")[0]
        )
        self.latent_transmat = jnp.array(
            self._make_env_transition(), device=jax.devices("cpu")[0]
        )
        self.generator = np.random.default_rng(42) # Reset generator for emissions, deterministic
        self.latent_emissions = jnp.array(
            self._make_env_emission(), device=jax.devices("cpu")[0]
        )

        self.hmm = CategoricalHMM(
            num_states=self.cfg.n_states, emission_dim=1, num_classes=self.cfg.n_obs
        )
        self.val_mode = False

    def to_device(self, device):
        self.index_to_latent = jax.device_put(
            self.index_to_latent, jax.devices(device)[0]
        )
        self.latent_transmat = jax.device_put(
            self.latent_transmat, jax.devices(device)[0]
        )
        self.latent_emissions = jax.device_put(
            self.latent_emissions, jax.devices(device)[0]
        )

    @partial(jax.jit, static_argnames="self")
    def get_transition(self, index):
        latent = self.index_to_latent[index]
        transition_latent = latent[: (3 + self.cfg.cycle_families + 2)]
        return self.latent_transmat[tuple(transition_latent)]

    @partial(jax.jit, static_argnames="self")
    def get_emission(self, index):
        latent = self.index_to_latent[index]
        emission_latent = latent[-(self.cfg.emission_groups + 1) :]
        return self.latent_emissions[tuple(emission_latent)]

    @partial(jax.jit, static_argnames="self")
    def get_startprobs(self, index):
        return jnp.ones(self.cfg.n_states) / self.cfg.n_states

    @partial(jax.jit, static_argnames="self")
    def filter(self, index, X, init=None):
        r"""Filter algorithm

        Args:
            index: Underlying HMM
            X: Sequence of observation

        Returns:
            log_likelihood: log_p(x_{1..t} | alpha), t \in [0,T]
            posterior: p(z_t | x_{1...t}, alpha), t \in [0,T] (NOTE: Includes p(z_0))

        """
        initial_probs = self.get_startprobs(index)
        emission_matrix = self.get_emission(index)
        log_likelihoods = jnp.log(emission_matrix[:, X].T)
        transition_matrix = self.get_transition(index)

        T, K = log_likelihoods.shape

        @jax.vmap
        def marginalize(m_ij, m_jk):
            A_ij_cond, lognorm = _condition_on(m_ij.A, m_jk.log_b)
            A_ik = A_ij_cond @ m_jk.A
            log_b_ik = m_ij.log_b + lognorm
            return FilterMessage(A=A_ik, log_b=log_b_ik)

        # Build initial messages
        if init is None:
            A0, log_b0 = _condition_on(initial_probs, log_likelihoods[0])
            A0 *= jnp.ones((K, K))
            log_b0 *= jnp.ones(K)
            A1T, log_b1T = jax.vmap(_condition_on, in_axes=(None, 0))(
                transition_matrix, log_likelihoods[1:]
            )
        else:
            A0, log_b0 = init[:-K], init[-K:]
            A0 = A0.reshape(K, K)

            A1T, log_b1T = jax.vmap(_condition_on, in_axes=(None, 0))(
                transition_matrix, log_likelihoods[0:]
            )
        initial_messages = FilterMessage(
            A=jnp.concatenate([A0[None, :, :], A1T]),
            log_b=jnp.vstack([log_b0, log_b1T]),
        )

        # Run the associative scan
        partial_messages = lax.associative_scan(marginalize, initial_messages)

        # Extract the marginal log likelihood and filtered probabilities (add p(z_0), p(x_0)=1)
        log_like = partial_messages.log_b[:, 0]
        z_post = partial_messages.A[:, 0, :]
        if init is None:
            log_like = jnp.concatenate([jnp.log(jnp.array([1.0])), log_like])
            z_post = jnp.concatenate([initial_probs[None], z_post])

        log_like = jnp.nan_to_num(log_like, nan=-jnp.inf)
        z_post = jnp.nan_to_num(z_post, nan=0.0)

        partial_messages = jnp.concatenate(
            [
                partial_messages.A[-T:].reshape(T, -1),
                partial_messages.log_b[-T:].reshape(T, -1),
            ],
            -1,
        )

        return log_like, z_post, partial_messages

    @partial(jax.jit, static_argnames="self")
    def bayesian_oracle(self, indices, X, initial_messages=False, log_alpha_prior=False):
        """Posterior predictive

        Args:
            indices (jnp.array): Environments considered possible
            X (jnp.array): Sequence of observations

        Returns:
            posterior_predictive: p(x_t | x_{<t}), t \in [1, T+1] (NOTE: INCLUDES p(x_1))
            posterior_latent: p(z_t | x)
        """

        # log_p(x_{1..t} | alpha)
        # p(z_t | x_{1...t}, alpha)
        if initial_messages is False:
            log_x_given_alpha, z_given_x_alpha, messages = jax.vmap(
                self.filter, (0, None)
            )(indices, X)
        else:
            log_x_given_alpha, z_given_x_alpha, messages = jax.vmap(
                self.filter, (0, None, 0)
            )(indices, X, initial_messages)

        # p(x_{t+1} | x_{1...t}, alpha) = sum_z p(x_{t+1} | z_{t+1}, alpha) p(z_{t+1} | z_t, alpha) p(z_t | x_{1...t}, alpha)
        log_x_given_x_alpha = jnp.log(
            jnp.einsum(
                "atz,azv,avx->atx",
                z_given_x_alpha,
                jax.vmap(self.get_transition)(indices),
                jax.vmap(self.get_emission)(indices),
            )
        )

        # Compute p(alpha)
        log_alpha = jnp.full(
            shape=(len(indices), 1), fill_value=jnp.log(1 / len(indices))
        ) * jnp.any(log_alpha_prior == False) + jnp.zeros((len(indices), 1)).at[
            :, 0
        ].set(
            log_alpha_prior
        ) * jnp.any(
            log_alpha_prior != False
        )
        # p(alpha | x_{1...t}) = p(x_{1...t} | alpha) p(alpha) / sum_{alpha} p(x_{<t} | alpha) p(alpha)
        log_alpha_given_x = log_x_given_alpha + jnp.broadcast_to(
            log_alpha, log_x_given_alpha.shape
        )
        log_alpha_given_x = (
            log_alpha_given_x - logsumexp(log_alpha_given_x, axis=0)[None]
        )

        # p(x_{t+1} | x_{1...t}) = \sum_{alpha} p(x_{t+1} | x_{1...t}, alpha) p(alpha | x_{1...t})
        x_given_x = jnp.nan_to_num(
            jnp.exp(
                logsumexp(log_x_given_x_alpha + log_alpha_given_x[..., None], axis=0)
            ),
            nan=0.0,
        )

        # p(z_t | x_{1...t}) = \sum_{alpha} p(z_t | x_{1...t}, alpha) p(alpha | x_{1...t})
        z_given_x = jnp.nan_to_num(
            jnp.exp(
                logsumexp(
                    jnp.log(z_given_x_alpha) + log_alpha_given_x[..., None], axis=0
                )
            ),
            nan=0.0,
        )

        return {
            "post_pred": x_given_x,
            "z_post": z_given_x,
            "log_alpha_post": log_alpha_given_x.T,
            "messages": messages.transpose(1, 0, 2),
        }

    def _make_env_transition(self):

        states = np.arange(self.cfg.n_states)

        # Generate base cycles
        base_transmat = np.zeros(
            shape=(
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
                self.cfg.n_states,
                self.cfg.n_states,
            ),
            dtype=np.float16,
        )
        for i in range(self.cfg.base_cycles):
            # The base cycle is an ordering of all the nodes
            base_cycle = self.generator.permutation(np.arange(len(states)))
            for j in range(self.cfg.base_directions):
                # Potentially reverse the direction of the cycle
                flipped_cycle = np.flip(base_cycle) if (j == 1) else base_cycle
                for k in range(self.cfg.base_speeds):
                    # Potentially accelate the speed at which the cycle is traversed
                    speed = k + 1
                    if gcd(speed, len(flipped_cycle)) == 1:
                        speed_cycle = [
                            flipped_cycle[(speed * m) % len(flipped_cycle)]
                            for m in range(len(flipped_cycle))
                        ]
                        base_transmat[i, j, k] = cycle_to_transmat(
                            speed_cycle, self.cfg.n_states
                        )
                    # Potentially this creates multiple non-overlapping cycles
                    else:
                        for l in range(gcd(speed, len(flipped_cycle))):
                            speed_cycle = [
                                flipped_cycle[(speed * m + l) % len(flipped_cycle)]
                                for m in range(len(flipped_cycle) // speed)
                            ]
                            base_transmat[i, j, k] += cycle_to_transmat(
                                speed_cycle, self.cfg.n_states
                            )

        # Generate cycle families
        family_transmat = np.zeros(
            shape=(
                self.cfg.cycle_families,
                self.cfg.group_per_family,
                self.cfg.family_directions,
                self.cfg.family_speeds,
                self.cfg.n_states,
                self.cfg.n_states,
            ),
            dtype=np.float16,
        )
        for i in range(self.cfg.cycle_families):
            for j in range(self.cfg.group_per_family):
                # Generate a group of cycle
                group = [
                    self.generator.choice(states, size=length, replace=False)
                    for length in self.generator.integers(
                        3, 9, size=self.cfg.cycle_per_group
                    )
                ]
                for k in range(self.cfg.family_directions):
                    # Potentially flip all the cycles in the group
                    flipped_group = [np.flip(c) for c in group] if (k == 1) else group
                    for l in range(self.cfg.family_speeds):
                        # Potentially accelerate the speed at which all the cycles are traversed
                        speed = l + 1
                        for c in flipped_group:
                            if gcd(speed, len(c)) == 1:
                                speed_cycle = [
                                    c[(speed * m) % len(c)]
                                    for m in range(
                                        len(c) // speed
                                        if gcd(speed, len(c)) != 1
                                        else len(c)
                                    )
                                ]
                                family_transmat[i, j, k, l] = cycle_to_transmat(
                                    speed_cycle, self.cfg.n_states
                                )
                            else:
                                for n in range(gcd(speed, len(c))):
                                    speed_cycle = [
                                        flipped_cycle[
                                            (speed * m + n) % len(flipped_cycle)
                                        ]
                                        for m in range(len(flipped_cycle) // speed)
                                    ]
                                    family_transmat[i, j, k, l] += cycle_to_transmat(
                                        speed_cycle, self.cfg.n_states
                                    )

        latents = (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
        )
        latent_transitions = np.zeros(
            shape=latents + [self.cfg.n_states, self.cfg.n_states], dtype=np.float16
        )

        for latent in product(*[range(n) for n in latents]):
            base_id, base_direction, base_speed = latent[0], latent[1], latent[2]
            family_ids = latent[3 : (3 + self.cfg.cycle_families)]
            group_direction, group_speed = latent[-2], latent[-1]

            # Add relevant cycles
            cycles = [base_transmat[base_id, base_direction, base_speed]] + [
                family_transmat[i, group_id, group_direction, group_speed]
                for (i, group_id) in enumerate(family_ids)
            ]
            transmat = np.stack(cycles).sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                transmat = transmat / transmat.sum(1)[:, None]
                transmat = np.nan_to_num(transmat, nan=0.0)

            latent_transitions[tuple(latent)] = transmat

        return latent_transitions

    def get_latents_shape(self):
        latents_shape = (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
            + [self.cfg.emission_group_size] * self.cfg.emission_groups
            + [self.cfg.emission_shifts]
        )
        return latents_shape

    def _make_env_emission(self):

        states = np.arange(self.cfg.n_states)
        obs = np.arange(self.cfg.n_obs)

        state_groups = np.array_split(states, self.cfg.emission_groups)
        emissions = np.zeros(
            shape=(
                self.cfg.emission_groups,
                self.cfg.emission_group_size,
                self.cfg.emission_shifts,
                self.cfg.n_states,
                self.cfg.n_obs,
            )
        )
        for i in range(self.cfg.emission_groups):
            group = list(state_groups[i])
            for j in range(self.cfg.emission_group_size):
                edges = preferential_attachement_edges(
                    group,
                    obs,
                    self.cfg.emission_edge_per_node * len(group),
                    generator=self.generator,
                )
                for k in range(self.cfg.emission_shifts):
                    # Shift starting edge within
                    for l in range(len(edges)):
                        source_idx = group.index(edges[l][0])
                        shifted_edge = (
                            group[(source_idx + k) % len(group)],
                            edges[l][1],
                        )
                        emissions[(i, j, k) + shifted_edge] = 1

        latents = [self.cfg.emission_group_size] * self.cfg.emission_groups + [
            self.cfg.emission_shifts
        ]
        latent_emissions = np.zeros(
            shape=latents + [self.cfg.n_states, self.cfg.n_obs], dtype=np.float16
        )
        for latent in product(*[range(n) for n in latents]):
            groups_id = latent[: self.cfg.emission_groups]
            emission_shift = latent[-1]

            emi = np.stack(
                [
                    emissions[group_id, emissions_id, emission_shift]
                    for (group_id, emissions_id) in enumerate(groups_id)
                ]
            ).sum(0)

            # Add a bit of noise
            emi = emi + self.cfg.emission_noise

            # Normalize
            with np.errstate(divide="ignore", invalid="ignore"):
                emi = emi / emi.sum(1)[:, None]
                emi = np.nan_to_num(emi, nan=0.0)

            latent_emissions[tuple(latent)] = emi

        return latent_emissions

    def _make_index_to_latent(self):

        latents = (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
            + [self.cfg.emission_group_size] * self.cfg.emission_groups
            + [self.cfg.emission_shifts]
        )
        index_to_latent = list(product(*[range(n) for n in latents]))
        index_to_latent = np.array(index_to_latent, dtype=np.int16)

        return index_to_latent

    def __len__(self):
        return len(self.index_to_latent)

    @property
    def latent_shape(self):
        return (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
            + [self.cfg.emission_group_size] * self.cfg.emission_groups
            + [self.cfg.emission_shifts]
        )

    @partial(jax.jit, static_argnames=["self", "n_steps", "reverse"])
    def sample(self, index, n_steps, key, initial_state=None, reverse=False):
        """Sample a sequence of observation from HMM <index>

        Args:
            index (int): ID of the HMM
            n_steps (int): Size of the sequence
            key (jr.PRNGKey): Jax randomness
            initial_state (int, optional): Starting state. Defaults to uniform over all states.
            reverse (bool, optional): Whether to sample in reverse (starting from the end, transition matrix transposed)

        Returns:
            X: Observations, jnp.array of size <n_steps>
            Y: States, jnp.array of size <n_steps>
        """
        # Handle None case before JIT: use uniform distribution if initial_state is None

        if initial_state is not None:
            startprobs = jax.nn.one_hot(initial_state, self.hmm.num_states)
        else:
            startprobs = self.get_startprobs(index)

        transitions = (
            self.get_transition(index).T if reverse else self.get_transition(index)
        )

        params = ParamsCategoricalHMM(
            initial=ParamsStandardHMMInitialState(startprobs),
            transitions=ParamsStandardHMMTransitions(transitions),
            emissions=ParamsCategoricalHMMEmissions(
                jnp.reshape(
                    self.get_emission(index),
                    shape=(self.cfg.n_states, 1, self.cfg.n_obs),
                )
            ),
        )

        Z, X = self.hmm.sample(params, key, n_steps)
        X = X[:, 0]

        if reverse:
            return jnp.flip(X), jnp.flip(Z)

        return X, Z

    def __getitem__(self, env):
        return self.__getitems__([env])

    def __getitems__(
        self,
        envs: List[int],
        seed: Optional[int] = None,
        length: Optional[Union[Tuple[int], int]] = None,
        intv_idx: Optional[Union[int, List[int], Tuple[int, int]]] = None,
        intv_envs: Optional[List[int]] = None,
    ):
        assert ~np.logical_xor(
            intv_idx is None, intv_envs is None
        ), "prefix_len and prefix_indices should both be set or neither be set"

        envs = jnp.array(envs)
        batch_size = len(envs)
        
        if intv_envs is None:
            intv_envs = jnp.array([], dtype=jnp.int32) 
        else:
            intv_envs = jnp.array(intv_envs)

        out_dict = {}

        # Set length if not set
        if length is None:
            length = self.cfg.context_length
        if isinstance(length, int):
            length = (length, length)

        variable_len = (length[0] != length[1]) & (not self.val_mode)

        # Set seed if not set
        if seed is None:
            seed = self.generator.integers(0, 1e10)

        # Sample sequences of maximum length
        seqs, states = jax.vmap(self.sample, (0, None, 0))(
            envs,
            length[1],
            jr.split(jr.PRNGKey(seed), batch_size),
        )

        seqs, states = j2t(seqs), j2t(states)

        out_dict.update({"input_ids": seqs, "states": states, "envs": envs, "intv_envs": intv_envs, "intv_idx": intv_idx})

        # Mid sequence intervention
        if intv_idx is not None:
            assert (
                variable_len is False
            ), "Cannot use variable length with interventions"
            # intv_idx: timestep in the sequence where the first intervened transition happens
            if isinstance(intv_idx, Iterable):
                intv_idx = self.generator.integers(
                    low=intv_idx[0], high=intv_idx[1], size=batch_size
                )
            else:
                intv_idx = np.full(shape=(batch_size,), fill_value=intv_idx)

            # Compute the state of the HMM at timestep <intv_idx -1>
            start_states = states[torch.arange(batch_size), intv_idx - 1]

            # Simulate the intervened HMM starting from this state
            # NOTE: Use deterministic seed based on intervention envs for reproducibility
            intv_seed = int(hash(tuple(np.asarray(intv_envs).tolist())) % (2**31))
            seqs_intv, states_intv = jax.vmap(self.sample, (0, None, 0, 0))(
                intv_envs,
                length[1] if length is not None else self.cfg.context_length[1],
                jr.split(jr.PRNGKey(intv_seed), batch_size),
                t2j(start_states),
            )
            # We remove the first observation because it came from state <intv_idx-1>
            seqs_intv, states_intv = j2t(seqs_intv)[:, 1:], j2t(states_intv)[:, 1:]

            raw_seqs = seqs.clone()
            raw_states = states.clone()

            for j in range(batch_size):
                # Intervene on the sequence
                seqs[j, intv_idx[j] :] = seqs_intv[j, : (seqs.shape[1] - intv_idx[j])]
                states[j, intv_idx[j] :] = states_intv[
                    j, : (states.shape[1] - intv_idx[j])
                ]

            out_dict.update({"intv_envs": intv_envs, "intv_idx": intv_idx})
            out_dict.update(
                {
                    "input_ids": seqs,
                    "states": states,
                    "raw_seqs": raw_seqs,
                    "raw_states": raw_states,
                }
            )

            return out_dict

        if variable_len:
            if self.cfg.adjust_varlen_batch:
                seqlens_ = self.generator.integers(
                    length[0],
                    length[1] + 1,
                    5 * batch_size,
                ).tolist()
                cu_seqlens_ = np.cumsum(seqlens_)
                seqlens = []
                for i in range(batch_size):
                    n_seqs = int(np.sum(cu_seqlens_ <= length[1]))
                    seqlens.append(seqlens_[:n_seqs])
                    if sum(seqlens[-1]) != length[1]:
                        seqlens[-1].append(
                            length[1] - np.sum(cu_seqlens_[n_seqs - 1]).item()
                        )

                    seqlens_ = seqlens_[n_seqs:]
                    cu_seqlens_ = (
                        cu_seqlens_[n_seqs:] - np.sum(cu_seqlens_[n_seqs - 1]).item()
                    )
                expanded_seqs = []
                for i in range(batch_size):
                    expanded_seqs.extend(
                        np.split(
                            seqs[i], indices_or_sections=np.cumsum(seqlens[i][:-1])
                        )
                    )
                seqs = expanded_seqs

                expanded_states = []
                for i in range(batch_size):
                    expanded_states.extend(
                        np.split(
                            states[i], indices_or_sections=np.cumsum(seqlens[i][:-1])
                        )
                    )
                states = expanded_states

                seqlens = torch.Tensor([len(seq) for seq in seqs])

                # Simply replace a random suffix length with padding
                ignore_mask = (
                    torch.arange(seqlens.max()).tile(len(seqs), 1) >= seqlens[:, None]
                )
                seqs = pad_sequence(
                    seqs,
                    batch_first=True,
                )
                states = pad_sequence(
                    states,
                    batch_first=True,
                )
                out_dict.update({"input_ids": seqs, "states": states, "ignore_mask": ignore_mask})
            else:
                seqlens = torch.Tensor(
                    self.generator.integers(
                        low=length[0], high=length[1] + 1, size=batch_size
                    )
                )

                ignore_mask = (
                    torch.arange(length[1]).tile(len(seqs), 1) >= seqlens[:, None]
                ).cuda()

                out_dict.update({"ignore_mask": ignore_mask})

        if self.cfg.start_at_n != None:
            assert isinstance(self.cfg.start_at_n, int)
            assert variable_len == False, "Cannot use <start_at_n> with variable length"
            ignore_mask = torch.arange(length[1]).tile(len(seqs), 1) < self.cfg.start_at_n
            out_dict.update({"ignore_mask": ignore_mask})

        return out_dict


# ============================================================================
# SubsetIntervened Dataset
# ============================================================================

class SubsetIntervened(Dataset):
    r"""
    Dataset of sequences where the underlying HMM undergoes an intervention during generation

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: MetaHMM
    prefix_indices: Sequence[int]

    def __init__(
        self,
        dataset: MetaHMM,
        prefix_indices: Sequence[int],
        suffix_indices: Sequence[int],
        intv_idx: Union[int, List[int], Tuple[int, int]],
    ) -> None:
        self.dataset = dataset
        self.prefix_indices = prefix_indices
        self.suffix_indices = suffix_indices
        self.intv_idx = intv_idx

    # Only support batched getitems like in the HMM dataset (for simplicity and efficiency)
    def __getitems__(self, indices: List[int]):
        return self.dataset.__getitems__(
            envs=[self.prefix_indices[idx] for idx in indices],
            intv_envs=[self.suffix_indices[idx] for idx in indices],
            intv_idx=self.intv_idx,
        )

    def __len__(self):
        return len(self.prefix_indices)


# ============================================================================
# MetaLearning Task Classes
# ============================================================================

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
        self.data = MetaHMM(cfg.data)
        self.data.to_device("cpu")

        # Build the train/validation set
        if self.cfg.val_size is not None:
            val_size = self.cfg.val_size
        elif self.cfg.val_ratio is not None:
            val_size = int(len(self.data) * self.cfg.val_ratio)
        else:
            raise Exception("Either val_size or val_ratio have to be defined")
        self.seen_tokens = torch.tensor(0).to("cuda")

        # Set numpy seed for deterministic train/val split
        np.random.seed(cfg.data.seed)
        val_latents = np.random.choice(
            len(self.data),
            val_size,
            replace=False,
        )
        train_latents = set(range(len(self.data)))
        train_latents.difference_update(val_latents)
        train_latents = np.array(list(train_latents))

        self.val_latents = torch.IntTensor(val_latents)
        self.train_latents = torch.IntTensor(train_latents)

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.data, indices=self.train_latents)
        self.val_data = Subset(self.data, indices=self.val_latents)

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
            shuffle=False,
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


class FSCLearningTask(MetaLearningTask):
    """Few-Shot Continual Learning Task with interventions"""

    def __init__(self, cfg: Optional[MetaLearningConfig] = None, **kwargs):

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(MetaLearningConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg
        metahmm_data = MetaHMM(cfg.data)
        metahmm_data.to_device("cpu")

        # Handle intv_idx configuration
        if self.cfg.intv_idx is None:
            # Default: use midpoint of context
            intv_idx = metahmm_data.cfg.context_length[0]//2
        elif isinstance(self.cfg.intv_idx, (list, tuple)):
            # Stochastic: keep as tuple for random sampling per batch
            intv_idx = tuple(self.cfg.intv_idx) if isinstance(self.cfg.intv_idx, list) else self.cfg.intv_idx
        else:
            # Fixed: use provided value
            intv_idx = self.cfg.intv_idx

        # Build the train/validation set
        if self.cfg.val_size is not None:
            val_size = self.cfg.val_size
        elif self.cfg.val_ratio is not None:
            val_size = int(len(metahmm_data) * self.cfg.val_ratio)
        else:
            raise Exception("Either val_size or val_ratio have to be defined")
        self.seen_tokens = torch.tensor(0).to("cuda")

        # Set numpy seed for deterministic train/val split
        np.random.seed(cfg.data.seed)
        val_latents = np.random.choice(
            len(metahmm_data),
            val_size,
            replace=False,
        )
        train_latents = set(range(len(metahmm_data)))
        train_latents.difference_update(val_latents)
        train_latents = np.array(list(train_latents))

        prefix_val_latents = val_latents[:len(val_latents)//2]
        suffix_val_latents = val_latents[len(val_latents)//2:]

        prefix_train_latents = train_latents[:len(train_latents)//2]
        suffix_train_latents = train_latents[len(train_latents)//2:]

        self.data = SubsetIntervened(dataset = metahmm_data,
                                     prefix_indices = np.concatenate([prefix_val_latents, prefix_train_latents]),
                                     suffix_indices = np.concatenate([suffix_val_latents, suffix_train_latents]),
                                     intv_idx = intv_idx)

        # Indices for Subset should be relative to SubsetIntervened, not the original dataset
        # Val data is at the beginning of the concatenated arrays
        self.val_latents = torch.arange(len(prefix_val_latents), dtype=torch.int32)
        # Train data comes after val data in the concatenated arrays
        self.train_latents = torch.arange(len(prefix_val_latents),
                                         len(prefix_val_latents) + len(prefix_train_latents),
                                         dtype=torch.int32)


# ============================================================================
# Evaluation Functions
# ============================================================================

def KLDiv(p, q):
    """Compute KL divergence between two distributions"""
    if isinstance(p, torch.Tensor):
        p = p
    else:
        p = torch.tensor(jax.device_get(p))

    if isinstance(q, torch.Tensor):
        q = q
    else:
        q = torch.tensor(jax.device_get(q))

    return torch.sum(p * (p.log() - q.log()), -1)


def NLL(p, X):
    """Compute negative log likelihood"""
    if isinstance(p, torch.Tensor):
        p = p
    else:
        p = torch.tensor(jax.device_get(p))

    if isinstance(X, torch.Tensor):
        X = X
    else:
        X = torch.tensor(jax.device_get(X))

    return torch.nn.functional.cross_entropy(
        input=torch.log(p[:, :-1].transpose(1, 2)),
        target=X[:, 1:].cpu().long(),
        reduction="none",
    )


def prepare_data(batch, rng_key):
    """Prepare data batch for metahmm evaluate function.

    Args:
        batch: Tuple from dataloader (shifted_idx, shifted_labels, envs, states, intv_envs, intv_idx)
        rng_key: JAX random key

    Returns:
        datap: Tuple of prepared data (xsp, ysp, envs, states, intv_envs, intv_idx)
        bkp: Backup data (None for this case)
        key: Updated JAX key
    """
    # Unpack batch tuple
    shifted_idx, shifted_labels, envs, states, intv_envs, intv_idx = batch

    # Prepare data tuple
    datap = (shifted_idx, shifted_labels, envs, states, intv_envs, intv_idx)
    bkp = None

    return datap, bkp, rng_key


def evaluate(model, dl, prepare_data, rng_key):
    """Computes the KL divergence between the model posterior predictive and the ground-truth

    Args:
        model: The model to evaluate
        dl: DataLoader for evaluation
        prepare_data: Function to prepare data batches
        rng_key: JAX random key

    Returns:
        dict: Dictionary with metrics including symKLDiv, ModelNLL, OracleNLL
    """


    def _run(model, x, bkp):
        """Run model inference and return softmax probabilities.

        Args:
            model: PyTorch model
            x: Input tensor (can be JAX array or PyTorch tensor)
            bkp: Backup data (unused)

        Returns:
            Softmax probabilities as numpy array
        """
        # Convert JAX array to PyTorch if needed
        if isinstance(x, jnp.ndarray):
            x_torch = j2t(x)
        elif isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x)
        else:
            x_torch = x

        # Ensure long type for token indices and move to CUDA
        x_torch = x_torch.long().cuda()

        # Run model with autocast for bfloat16 compatibility
        with torch.no_grad():
            # Use autocast to ensure compatibility with flash attention
            device_type = "cuda" if x_torch.is_cuda else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                outputs = model(x_torch)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

            # Apply softmax (outside autocast to ensure float32 for numerical stability)
            #probs = torch.nn.functional.softmax(logits.float(), dim=-1)
            probs = torch.nn.functional.softmax(logits.float(), dim=-1)


        # Convert back to numpy for compatibility with JAX code
        return probs.cpu().numpy()

    # Get the MetaHMM dataset from the dataloader
    if hasattr(dl.dataset.dataset, "bayesian_oracle"):
        metahmm = dl.dataset.dataset
    else:
        metahmm = dl.dataset.dataset.dataset

    metrics = {
        "symKLDiv": [],
        "ModelNLL": [],
        "OracleNLL": []
    }

    dl_iter = iter(dl)
    for i in tqdm(range(0, len(dl)), leave=False):
        batch = next(dl_iter)
        datap, bkp, key = prepare_data(batch, rng_key)

        xsp, ysp, envs, states, intv_envs, intv_idx = datap

        # Convert to JAX arrays if needed
        x_jax = t2j(xsp)
        x_torch = j2t(xsp)

        # Compute the model's posterior predictive
        model_pp = _run(model, x_torch, bkp)

        oracle_pp = []
        if intv_idx is None or intv_envs is None:
            # No intervention: standard oracle computation
            for j, _x in enumerate(x_jax):  # loop over batch
                assumed_envs = t2j(envs[j][None])
                oracle_pp.append(metahmm.bayesian_oracle(assumed_envs, _x)["post_pred"])
        else:
            # With intervention: use the actual intervention point for each sequence
            # Convert intv_idx to numpy array for indexing
            if isinstance(intv_idx, torch.Tensor):
                intv_idx_arr = intv_idx.cpu().numpy()
            elif isinstance(intv_idx, jnp.ndarray):
                intv_idx_arr = np.array(intv_idx)
            else:
                intv_idx_arr = np.array(intv_idx) if not isinstance(intv_idx, np.ndarray) else intv_idx

            for j, _x in enumerate(x):  # loop over batch
                prefix_envs = t2j(envs[j][None])
                suffix_envs = t2j(intv_envs[j][None])
                # Note: x is shifted (corresponds to input_ids[:-1]), so intv_idx needs adjustment
                # The intervention happens at intv_idx in the original sequence, which is intv_idx-1 in shifted x
                # Use the actual intervention point that was used to generate this sequence
                intv_point = int(intv_idx_arr[j])
                # Prefix: compute oracle on observations before intervention
                prefix_pp = metahmm.bayesian_oracle(prefix_envs, _x)["post_pred"][:intv_point]
                # Suffix: compute oracle on observations from intervention point onward
                suffix_pp = metahmm.bayesian_oracle(suffix_envs, _x)["post_pred"][intv_point:]
                # Concatenate without overlap
                oracle_pp.append(jnp.concatenate([prefix_pp, suffix_pp]))

        oracle_pp = jnp.stack(oracle_pp)[:, 1:, : metahmm.cfg.n_obs]

        # Compute forward/backward KL and NLL of model and oracle
        f_kl = KLDiv(oracle_pp, model_pp)
        b_kl = KLDiv(model_pp, oracle_pp)
        model_nll = NLL(model_pp, x_jax)
        oracle_nll = NLL(oracle_pp, x_jax)

        metrics["symKLDiv"].append(0.5 * (f_kl + b_kl))
        metrics["ModelNLL"].append(model_nll)
        metrics["OracleNLL"].append(oracle_nll)

    return {k: np.nan_to_num(np.row_stack(v), posinf=0) for k, v in metrics.items()}


# ============================================================================
# Wrapper Functions for dataset_wrappers.py compatibility
# ============================================================================

def metalearning_hmm(batch_size,
                     val_size=None,
                     val_ratio=0.1,
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
    - val_size: int or None, number of validation HMM environments (overrides val_ratio if set)
    - val_ratio: float, ratio of total HMMs to use for validation (default: 0.1)
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
        val_size=val_size,
        val_ratio=val_ratio,
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


def fsc_metalearning_hmm(batch_size,
                         val_size=None,
                         val_ratio=0.1,
                         use_prepare_step=True,
                         drop_last=True,
                         intv_idx=None,
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
    Few-Shot Continual (FSC) Meta-learning Hidden Markov Model dataset with DataLoader wrapper from Gagnon et al.

    This function wraps the FSCLearningTask class which provides PyTorch DataLoaders for the FSC variant
    of the MetaHMM task. Unlike the standard MetaLearning task, FSC splits HMM environments into prefix
    and suffix groups and performs mid-sequence interventions where the underlying HMM switches.

    **Arguments:**
    - batch_size: int, the batch size for the dataloaders
    - val_size: int or None, number of validation HMM environments (overrides val_ratio if set)
    - val_ratio: float, ratio of total HMMs to use for validation (default: 0.1)
    - use_prepare_step: bool, if True returns preprocessed batches (shifted_idx, shifted_labels, envs, states),
                       if False returns raw batch dicts (default: True)
    - drop_last: bool, if True drops the last incomplete batch to ensure all batches have the same size.
                Important when using data parallelism/sharding (default: True)
    - intv_idx: int, list, tuple, or None, intervention timestep where HMM switches.
                If int: fixed intervention point for all sequences
                If list/tuple [min, max]: stochastically samples intervention point uniformly from [min, max) per sequence
                If None: defaults to context_length[0]//2 (midpoint)
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
    - dl_kwargs: dict, additional keyword arguments (not used for FSCLearningTask)

    **Returns:**
    - dataloader_train: PyTorch DataLoader for training
    - dataloader_test: PyTorch DataLoader for testing (same as validation)
    - dataloader_val: PyTorch DataLoader for validation
    - input_size: int, vocabulary size (n_obs)
    - output_size: int, vocabulary size (n_obs)

    **Usage:**
    When use_prepare_step=True (default):
        shifted_idx, shifted_labels, envs, states = next(iter(dataloader_train))
        # Note: sequences contain interventions where the HMM switches mid-sequence
        # The ignore_mask in shifted_labels indicates which positions to train on

    When use_prepare_step=False:
        batch_dict = next(iter(dataloader_train))
        # batch_dict contains: 'input_ids', 'states', 'envs', 'raw_seqs', 'raw_states', 'ignore_mask'
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
        val_size=val_size,
        val_ratio=val_ratio,
        intv_idx=intv_idx,
    )

    # Create FSC task and setup dataloaders
    task = FSCLearningTask(metalearning_config)
    task.setup()

    # Get the dataloaders with optional prepare_step and drop_last
    dataloader_train = task.train_dataloader(use_prepare_step=use_prepare_step, drop_last=drop_last)
    dataloader_val = task.val_dataloader(use_prepare_step=use_prepare_step, drop_last=drop_last)
    dataloader_test = dataloader_val  # Use validation for test as well

    # Return dataloaders and input/output sizes
    return dataloader_train, dataloader_test, dataloader_val, n_obs, n_obs





# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Example 1: Standard MetaHMM")
    print("=" * 80)

    # Create dataloaders using wrapper function
    train_dl, test_dl, val_dl, input_size, output_size = metalearning_hmm(
        batch_size=256,
        val_size=1000,
        n_states=30,
        n_obs=60,
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

    # Create dataloaders with interventions
    train_dl_fsc, test_dl_fsc, val_dl_fsc, input_size, output_size = fsc_metalearning_hmm(
        batch_size=256,
        val_size=1000,
        intv_idx=[100, 300],  # Random intervention between timesteps 100-300
        n_states=30,
        n_obs=60,
        context_length=(400, 400),
        seed=42
    )

    print(f"Input size: {input_size}, Output size: {output_size}")
    batch = next(iter(train_dl_fsc))
    print(f"Batch is a tuple with {len(batch)} elements")
    print(f"  shifted_idx shape: {batch[0].shape}")
    print(f"  shifted_labels shape: {batch[1].shape}")
    print(f"  envs_torch shape: {batch[2].shape}")
    print(f"  states shape: {batch[3].shape}")
    if batch[4] is not None:
        print(f"  intv_envs shape: {batch[4].shape}")
    if batch[5] is not None:
        print(f"  intv_idx shape: {batch[5].shape}")

    print("\n✓ All examples completed successfully!")
