from collections.abc import Sequence
import enum
import functools
import inspect
import itertools
import logging
import os
import re
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import numpy as np
from jax.experimental import multihost_utils
import jax
from jax import random

import t5.data

from t5x.examples.t5 import network
import t5x
from t5x import models
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
from t5x.infer import _extract_tokens_and_aux_values
from t5x.infer import _Inferences
from t5x.interactive_model import InteractiveModel
from t5x.interactive_model import get_batches_from_seqio
from t5x.interactive_model import get_dataset_from_natural_text_examples
from t5x.interactive_model import get_gin_config_from_interactive_model
from t5x.interactive_model import T5XScriptType
from t5x.interactive_model import InferenceType

t5_config = network.T5Config(
    vocab_size=32128,
    dtype="bfloat16",
    emb_dim=4096,
    num_heads=64,
    num_encoder_layers=24,
    num_decoder_layers=0,
    head_dim=64,
    mlp_dim=10240,
    mlp_activations=("gelu", "linear"),
    dropout_rate=0.0,
    logits_via_embedding=False,
)
module = network.Transformer(config=t5_config)
model = t5x.models.EncoderDecoderModel(
    module=module,
    input_vocabulary=t5.data.get_default_vocabulary(),
    output_vocabulary=t5.data.get_default_vocabulary(),
    optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
)
# Define checkpoint arguments.
checkpoint_path = "gs://t5-data/pretrained_models/t5.1.1.xxl/model.ckpt-1000000"
dtype = "bfloat16"

interactive_model = InteractiveModel(
    batch_size=8,
    task_feature_lengths={"inputs": 512, "targets": 0},
    output_dir="/tmp/output_dir",
    partitioner=partitioning.PjitPartitioner(
        num_partitions=1,
        model_parallel_submesh=None,
        logical_axis_rules=partitioning.standard_logical_axis_rules(),
    ),
    model=model,
    dtype="bfloat16",
    restore_mode="specific",
    checkpoint_path=checkpoint_path,
    input_shapes={
        # "encoder_input_tokens": np.array([8, 38]),
        # "decoder_target_tokens": np.array([8, 18]),
        # "decoder_input_tokens": np.array([8, 18]),
        # "decoder_loss_weights": np.array([8, 18]),
        "encoder_input_tokens": np.array([8, 512]),
        "decoder_target_tokens": np.array([8, 0]),
        "decoder_input_tokens": np.array([8, 0]),
        "decoder_loss_weights": np.array([8, 0]),
    },
    input_types=None,
)