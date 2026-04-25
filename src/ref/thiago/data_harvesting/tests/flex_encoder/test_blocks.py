import torch
from torch import nn

from data_harvesting.encoder.blocks import CentralizedAgentBlock, PerAgentBlock, SharedAgentBlock


class _RecordingSeqHead(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(2, out_dim, bias=False)
        self.last_x: torch.Tensor | None = None
        self.last_agent_idx: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, agent_idx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self.last_x = x.detach().clone()
        self.last_agent_idx = agent_idx.detach().clone()
        self.last_mask = None if mask is None else mask.detach().clone()
        return self.proj(x.mean(dim=-2))


class _RecordingFlatHead(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(2, out_dim, bias=False)
        self.last_x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x.detach().clone()
        return self.proj(x)


class _LinearMix(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_centralized_block_shapes_and_mask_routing() -> None:
    torch.manual_seed(0)

    drones_head = _RecordingSeqHead(out_dim=3)
    sensors_head = _RecordingSeqHead(out_dim=3)
    block = CentralizedAgentBlock(
        seq_heads={"drones": drones_head, "sensors": sensors_head},
        mix_layer=_LinearMix(in_dim=6, out_dim=4),
        n_agents=3,
    )

    observation = {
        "drones": torch.randn(2, 3, 4, 2),
        "sensors": torch.randn(2, 3, 4, 2),
    }
    mask = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool)

    output = block(observation, mask)

    assert tuple(output.shape) == (2, 4)

    assert drones_head.last_x is not None
    assert drones_head.last_agent_idx is not None
    assert drones_head.last_mask is not None

    assert tuple(drones_head.last_x.shape) == (2, 12, 2)
    assert tuple(drones_head.last_agent_idx.shape) == (2, 12, 1)
    assert tuple(drones_head.last_mask.shape) == (2, 12)

    expected_agent_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    assert torch.equal(drones_head.last_agent_idx[0, :, 0], expected_agent_idx)

    expected_mask_row0 = torch.tensor(
        [True, True, True, True, False, False, False, False, True, True, True, True],
        dtype=torch.bool,
    )
    assert torch.equal(drones_head.last_mask[0], expected_mask_row0)


def test_shared_block_output_shape_and_input_routing_without_mask() -> None:
    torch.manual_seed(0)

    seq_head = _RecordingSeqHead(out_dim=3)
    flat_head = _RecordingFlatHead(out_dim=2)
    block = SharedAgentBlock(
        seq_heads={"drones": seq_head},
        flat_heads={"agent_id": flat_head},
        mix_layer=_LinearMix(in_dim=5, out_dim=4),
        n_agents=3,
    )

    observation = {
        "drones": torch.randn(2, 3, 4, 2),
        "agent_id": torch.randn(2, 3, 2),
    }

    output = block(observation, mask=None)

    assert tuple(output.shape) == (2, 3, 4)

    assert seq_head.last_x is not None
    assert seq_head.last_agent_idx is not None
    assert seq_head.last_mask is None
    assert tuple(seq_head.last_x.shape) == (6, 4, 2)
    assert tuple(seq_head.last_agent_idx.shape) == (6, 1)

    expected_agent_idx = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    assert torch.equal(seq_head.last_agent_idx[:, 0], expected_agent_idx)

    assert flat_head.last_x is not None
    assert tuple(flat_head.last_x.shape) == (2, 3, 2)


def test_per_agent_block_selects_agent_specific_inputs_and_idx_values() -> None:
    torch.manual_seed(0)

    seq_head = _RecordingSeqHead(out_dim=3)
    flat_head = _RecordingFlatHead(out_dim=2)
    block = PerAgentBlock(
        seq_heads={"drones": seq_head},
        flat_heads={"agent_id": flat_head},
        mix_layer=_LinearMix(in_dim=5, out_dim=4),
    )

    observation = {
        "drones": torch.randn(2, 3, 4, 2),
        "agent_id": torch.randn(2, 3, 2),
    }

    for agent_idx in range(3):
        output = block(observation, agent_idx=agent_idx, mask=None)

        assert tuple(output.shape) == (2, 4)

        assert seq_head.last_x is not None
        assert seq_head.last_agent_idx is not None
        assert seq_head.last_mask is None
        assert tuple(seq_head.last_x.shape) == (2, 4, 2)
        assert torch.allclose(seq_head.last_x, observation["drones"][:, agent_idx])

        assert tuple(seq_head.last_agent_idx.shape) == (2, 1, 1)
        expected_agent_idx = torch.full((2,), agent_idx, dtype=seq_head.last_agent_idx.dtype)
        assert torch.equal(seq_head.last_agent_idx[:, 0, 0], expected_agent_idx)

        assert flat_head.last_x is not None
        assert tuple(flat_head.last_x.shape) == (2, 2)
        assert torch.allclose(flat_head.last_x, observation["agent_id"][:, agent_idx])
