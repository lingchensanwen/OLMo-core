"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/deepseek/train.py run_name [OVERRIDES...]
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.nn.rope import RoPELlamaScalingConfig, RoPELinearScalingConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.from_hf("deepseek-ai/deepseek-coder-1.3b-base")

    model_config = TransformerConfig.deepseek_1B(
        vocab_size=tokenizer_config.vocab_size,  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        fused_ops=False,
        use_flash=False,
        rope_scaling=RoPELinearScalingConfig(factor=4.0),
        # dp_config=TransformerDataParallelConfig(
        #     name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        # ),
    )

    optim_config = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    dataset_config = NumpyDatasetConfig.glob(
        "/scratch/07144/yw23374/data/astro_cpt_dscoder/test//tokens/part-0-00000.npy",  # can be globs
        name=NumpyDatasetType.fsl,
        sequence_length=8192,
        max_target_sequence_length=8192,
        #  name=NumpyDatasetType.vsl,
        #  max_sequence_length=2048,
        #  min_sequence_length=256,
        #  vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=4),
        tokenizer=tokenizer_config,
        work_dir="/tmp/dataset-cache",
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,
        seed=0,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            rank_microbatch_size=16 * 1024,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            load_key_mapping={
                # For backwards compatibility when loading older checkpoints.
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(
                min_sequence_length=128, warmup_steps=100, enabled=False
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=20,
                ephemeral_save_interval=10,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig(
                    paths=["/scratch/07144/yw23374/data/astro_cpt_dscoder/test//tokens/part-0-00000.npy"],
                    metadata=[{"label": "cpt-validation"}],
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=8192,
                    tokenizer=tokenizer_config,
                    work_dir="/tmp/dataset-cache",
                ),
                eval_interval=10,
                eval_duration=Duration.steps(10),
            ),
        )
        # .with_callback(
        #     "downstream_evaluator",
        #     DownstreamEvaluatorCallbackConfig(
        #         tasks=["hellaswag"],
        #         tokenizer=tokenizer_config,
        #         eval_interval=250,
        #     ),
        # )
    )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    device = get_default_device()

    # Build the world mesh, if needed.
    world_mesh = config.model.build_mesh(device=device)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, mesh=world_mesh)
    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()
