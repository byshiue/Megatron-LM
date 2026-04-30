# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


@pytest.mark.parametrize(
    # ("tp_size", "sp", "cp_size"),
    # [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size):
        # Initialize parallel and random seed
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        # Initialize model
        self.transformer_config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=self.transformer_config
        ).submodules

        self.gdn = GatedDeltaNet(
            self.transformer_config,
            submodules=gdn_submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        self.gdn = self.gdn.cuda().half()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.half,
        )
        attention_mask = None

        output, bias = gdn(hidden_states, attention_mask)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert output.shape[0] == seq_length // self.sp_size // self.cp_size, (
            f"Output shape {output.shape[0]=} mismatch with "
            f" {seq_length=} // {self.sp_size=} // {self.cp_size=}."
        )
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == gdn.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {gdn.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // gdn.cp_size

        qkv_last_dim = (2 * gdn.qk_dim_local_tp + gdn.v_dim_local_tp) // gdn.cp_size
        qkv = torch.randn(
            batch, seq_len, qkv_last_dim, device=torch.cuda.current_device(), dtype=torch.half
        )
        gate = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn.value_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.half,
        )
        beta = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.half,
        )
        alpha = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.half,
        )

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        with torch._dynamo.config.patch(disable=True):
            query, key, value, gate_out, beta_out, alpha_out = (
                gdn._prepare_qkv_for_gated_delta_rule(qkv, gate, beta, alpha, batch, seq_len)
            )

        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        assert query.is_contiguous()
        assert key.is_contiguous()
        assert value.is_contiguous()

        A_log_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.half
        )
        dt_bias_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.half
        )

        with torch._dynamo.config.patch(disable=True):
            g, beta_sig = gdn._compute_g_and_beta(A_log_mock, dt_bias_mock, alpha, beta)

        assert g.dtype == torch.float32
        assert g.shape == alpha.shape
        assert beta_sig.shape == beta.shape


@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        # (4, False, 1),  # TP w/o SP
        # (4, True, 1),  # TP w/ SP
        # (1, False, 2),  # CP
        # (2, False, 2),  # TP w/o SP + CP
        # (2, True, 2),  # TP w/ SP + CP
        (1, False, 1),  #
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(tmp_path_dist_ckpt, tp, sp, cp, dtype):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=64,
        activation_func=F.silu,
        # fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-3, 1e-3

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=8192,
        micro_batch_size=2,
        model_dtype=dtype,
    )

    _benchmark_gated_delta_net_bwd(dtype=dtype)


def _bench(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats  # ms


def _benchmark_gated_delta_net_bwd(
    dtype: torch.dtype = torch.float16,
    seq_lengths: list = None,
    warmup: int = 5,
    repeats: int = 20,
):
    """E2E benchmark: full forward+backward of GatedDeltaNet.

    Compares different CUDA kernel configurations against the Triton baseline.
    Uses the same model config as test_parallel_gated_delta_net_correctness.
    """
    import os as _os

    if seq_lengths is None:
        seq_lengths = [8192]
        # seq_lengths = [1024, 2048, 4096, 8192]

    dtype_str = "fp16" if dtype == torch.float16 else "bf16"

    # Kernel configs to compare (name -> env var overrides)
    configs = [
        ("Triton (baseline)",        {}),
        ("CUDA: wy_bwd",             {"FLA_CUTE_WY_BWD": "1"}),
        ("CUDA: delta_h",            {"FLA_CUTE_BWD_DHU": "1"}),
        ("CUDA: dqkwg",              {"FLA_CUTE_BWD_DQKWG": "1"}),
        ("CUDA: dhu+dqkwg",          {"FLA_CUTE_BWD_DHU_DQKWG": "1"}),
        ("CUDA: dhu+dqkwg kernel",   {"FLA_CUTE_BWD_DHU_DQKWG_KERNEL": "1"}),
        ("CUDA: all three",          {"FLA_CUTE_WY_BWD": "1",
                                        "FLA_CUTE_BWD_DHU": "1",
                                        "FLA_CUTE_BWD_DQKWG": "1"}),
        ("CUDA: wy+dhu+dqkwg",       {"FLA_CUTE_WY_BWD": "1",
                                        "FLA_CUTE_BWD_DHU_DQKWG": "1"}),
    ]
    if _os.environ.get("MCORE_GDN_BENCH_BASELINE_ONLY", "0") == "1":
        configs = configs[:1]

    # Env vars that control kernel dispatch
    _all_flags = [
        "FLA_CUTE_WY_BWD",
        "FLA_CUTE_BWD_DHU",
        "FLA_CUTE_BWD_DQKWG",
        "FLA_CUTE_BWD_DHU_DQKWG",
        "FLA_CUTE_BWD_DHU_DQKWG_KERNEL",
    ]

    def _set_env(overrides):
        for flag in _all_flags:
            _os.environ.pop(flag, None)
        for k, v in overrides.items():
            _os.environ[k] = v

    # Build column widths
    col_w = max(len(c[0]) for c in configs) + 2
    sep = "=" * (16 + col_w * len(configs) + 2)

    print(f"\n{sep}")
    print(f"  E2E GatedDeltaNet forward+backward  [dtype={dtype_str}]")
    print(f"  Model: hidden=128  K=128  V=128  num_kv_heads=64/64  B=2")
    print(sep)

    # Header row
    hdr = f"{'T':<16}"
    for name, _ in configs:
        hdr += f"  {name:>{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    # _test_parallel_attention_correctness already called destroy_model_parallel.
    # Re-initialize a single-rank parallel state so GatedDeltaNet linear layers work.
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)

    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        cfg = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=64,
            linear_num_value_heads=64,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=64,
            activation_func=F.silu,
            fp16=(dtype == torch.float16),
            bf16=(dtype == torch.bfloat16),
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        submodules = get_experimental_attention_variant_module_spec(
            config=cfg
        ).submodules
        # Build one model and reuse it across all configs and T values
        _set_env({})
        gdn = GatedDeltaNet(
            cfg, submodules=submodules, layer_number=1,
            bias=False, conv_bias=False, conv_init=1.0,
            use_qk_l2norm=True, A_init_range=(1, 16),
            pg_collection=pg_collection,
        ).cuda().to(dtype)
        gdn.eval()

        B = 2
        for T in seq_lengths:
            x = torch.randn(T, B, cfg.hidden_size, device="cuda", dtype=dtype)

            times = []
            for _name, env_overrides in configs:
                _set_env(env_overrides)
                # Sanitize config name into a valid NVTX range label
                nvtx_label = f"T={T}/{dtype_str}/{_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}"

                def fwd_bwd(model=gdn, inp=x, label=nvtx_label):
                    inp = inp.detach().requires_grad_(True)
                    with torch.cuda.nvtx.range(label):
                        out, _ = model(inp, attention_mask=None)
                        out.sum().backward()
                    torch.cuda.synchronize()

                with torch.cuda.nvtx.range(f"bench/{nvtx_label}"):
                    ms = _bench(fwd_bwd, warmup=warmup, repeats=repeats)
                times.append(ms)

            # Print row: absolute time for baseline, time + speedup for CUDA configs
            baseline_ms = times[0]
            line = f"T={T:<13}"
            for i, (ms, (name, _)) in enumerate(zip(times, configs)):
                if i == 0:
                    line += f"  {ms:>{col_w - 2}.3f}ms"
                else:
                    speedup = baseline_ms / ms
                    line += f"  {ms:>{col_w - 9}.3f}ms({speedup:+.2f}x)"
            print(line)

        del gdn
    finally:
        # Restore clean env and tear down parallel state
        _set_env({})
        Utils.destroy_model_parallel()

    print(f"\n  (ms/iter, warmup={warmup}, repeats={repeats}; speedup vs Triton baseline)\n")
