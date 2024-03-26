import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core.distributed.fsdp import FSDP, FSDPDebugConfig
from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter

from ..utils import BACKENDS, get_default_device, run_distributed_test


def run_fsdp_against_non_distributed_model(model_factory, model_data_factory):
    """
    Compare outputs from forward pass and gradients to those from a non-distributed model.
    """
    model_data = model_data_factory().to(get_default_device())

    model = model_factory()
    fsdp1 = FSDP(model_factory(), _debug_config=FSDPDebugConfig(no_reduce_grads=True))
    fsdp2 = FSDP(model_factory())

    # Ensure params for all models on all ranks match.
    for param in model.parameters():
        with torch.no_grad():
            dist.broadcast(param.data, 0)

    for fsdp in (fsdp1, fsdp2):
        with fsdp.summon_full_params():
            fsdp.module.load_state_dict(model.state_dict())

        for name, param in fsdp1.module.named_parameters():
            assert isinstance(param, ShardedFlatParameter)
            assert param.is_sharded
            assert param.grad is None
            with torch.no_grad():
                torch.testing.assert_close(param.data, param.sharded_chunk(model.state_dict()[name]))

    # Run forward/backward pass on non-distributed model and collect grads for comparison.
    expected_grads = {}
    loss = model(model_data).sum()
    loss.backward()
    for param_name, param in model.named_parameters():
        expected_grads[param_name] = param.grad.detach()

    # Run forward pass on FSDP models.
    fsdp1_loss = fsdp1(model_data).sum()
    fsdp2_loss = fsdp2(model_data).sum()

    with torch.no_grad():
        torch.testing.assert_close(loss, fsdp1_loss)
        torch.testing.assert_close(loss, fsdp2_loss)

    # Models should be in a sharded state again.
    for fsdp in (fsdp1, fsdp2):
        for param in fsdp.parameters():
            assert isinstance(param, ShardedFlatParameter)
            assert param.is_sharded

    # Trigger backward pass on FSDP model.
    fsdp1_loss.backward()
    fsdp2_loss.backward()

    # Check gradients and ensure model is in a sharded state again.
    for name, param in fsdp1.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        with torch.no_grad():
            torch.testing.assert_close(
                param.grad, expected_grads[name], msg=lambda m: f"On gradient for '{name}'. {m}"
            )

    # Now manually reduce grads for the 1st FSDP model to compare to the 2nd FSDP model.
    for (name, param1), param2 in zip(fsdp1.module.named_parameters(), fsdp2.module.parameters()):
        with torch.no_grad():
            dist.all_reduce(param1.grad, group=fsdp1.process_group)
            torch.testing.assert_close(
                param2.sharded_chunk(param1.grad), param2.grad, msg=lambda m: f"On gradient for '{name}'. {m}"
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_against_non_distributed_model(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_fsdp_against_non_distributed_model,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )


def run_fsdp_against_ddp(model_factory, model_data_factory):
    """
    Compare outputs from forward pass and gradients against those from a DDP model.
    """
    model_data = model_data_factory().to(get_default_device())

    ddp_model = DDP(model_factory())
    fsdp_model = FSDP(model_factory())

    with fsdp_model.summon_full_params():
        fsdp_model.module.load_state_dict(ddp_model.module.state_dict())

    for name, param in fsdp_model.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is None
        with torch.no_grad():
            torch.testing.assert_close(param.data, param.sharded_chunk(ddp_model.module.state_dict()[name]))

    # Run forward/backward pass on DDP model and collect grads for comparison.
    ddp_loss = ddp_model(model_data).sum()
    ddp_loss.backward()
    expected_grads = {}
    for param_name, param in ddp_model.module.named_parameters():
        expected_grads[param_name] = param.grad.detach()

    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Run forward pass on FSDP model.
    fsdp_loss = fsdp_model(model_data).sum()

    with torch.no_grad():
        torch.testing.assert_close(ddp_loss, fsdp_loss)

    # Model should be in a sharded state again.
    for param in fsdp_model.parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded

    # Trigger backward pass on FSDP model.
    fsdp_loss.backward()

    # Model should be in a sharded state again.
    for name, param in fsdp_model.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        with torch.no_grad():
            # NOTE: DDP *averages* gradients over ranks, FSDP just takes the sum.
            torch.testing.assert_close(
                param.grad,
                param.sharded_chunk(expected_grads[name] * dist.get_world_size()),
                msg=lambda m: f"On gradient for '{name}'. {m}",
            )

    # Run optimizer step.
    optim.step()


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_against_ddp(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_fsdp_against_ddp,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )
