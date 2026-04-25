import torch
from torch import nn

from data_harvesting.encoder import (
    SequentialEncoder,
    SequentialEncoderConfig,
    SequentialEncoderInput,
)


def _make_sequential_head(*, agentic_encoding: bool = False) -> SequentialEncoder:
    return SequentialEncoder(
        input=SequentialEncoderInput(key="drones", input_size=2),
        config=SequentialEncoderConfig(
            embed_dim=16,
            head_dim=8,
            num_heads=2,
            ff_dim=32,
            depth=1,
            dropout=0.0,
            max_num_agents=4,
            agentic_encoding=agentic_encoding,
        ),
        device=torch.device("cpu"),
    )


class _ZeroObsEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*x.shape[:-1], self.embed_dim), dtype=x.dtype, device=x.device)


class _IndexAgentEmbedder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(torch.float32)
        return idx.unsqueeze(-1).expand(*idx.shape, self.embed_dim)


class _CaptureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.last_input = x.detach().clone()
        self.last_mask = None if src_key_padding_mask is None else src_key_padding_mask.detach().clone()
        return x


class _MaskedValueTransformer(nn.Module):
    def __init__(self, valid_value: float, masked_value: float):
        super().__init__()
        self.valid_value = valid_value
        self.masked_value = masked_value

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert src_key_padding_mask is not None
        output = torch.full_like(x, self.valid_value)
        output[src_key_padding_mask] = self.masked_value
        return output


def test_sequential_head_mask_argument_changes_forward_when_elements_masked() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 5, 2)
    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out_none = head(x, agent_idx, None)
    out_masked = head(x, agent_idx, mask)
    out_all_true = head(x, agent_idx, torch.ones_like(mask))

    assert not torch.allclose(out_none, out_masked, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_none, out_all_true, atol=1e-6, rtol=1e-6)


def test_sequential_head_mask_argument_changes_backward_when_elements_masked() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x_nomask = torch.randn(2, 5, 2, requires_grad=True)
    x_masked = x_nomask.detach().clone().requires_grad_(True)

    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out_nomask = head(x_nomask, agent_idx, None)
    out_masked = head(x_masked, agent_idx, mask)

    out_nomask.sum().backward()
    out_masked.sum().backward()

    assert x_nomask.grad is not None
    assert x_masked.grad is not None
    assert torch.allclose(x_masked.grad[~mask], torch.zeros_like(x_masked.grad[~mask]), atol=1e-8, rtol=0)
    assert torch.max(torch.abs(x_nomask.grad[~mask])) > 0


def test_sequential_head_masked_positions_receive_zero_gradients() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 5, 2, requires_grad=True)
    mask = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out = head(x, agent_idx, mask)
    out.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad[~mask], torch.zeros_like(x.grad[~mask]), atol=1e-8, rtol=0)
    assert torch.count_nonzero(x.grad[mask]) > 0


def test_sequential_head_output_shape_preserves_single_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(4, 5, 2)
    agent_idx = torch.zeros((4, 1), dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (4, 16)
    assert out.device == x.device


def test_sequential_head_output_shape_preserves_multi_batch_dims() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(2, 3, 5, 2)
    agent_idx = torch.zeros((2, 3, 1), dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (2, 3, 16)
    assert out.device == x.device


def test_sequential_head_output_shape_preserves_no_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head()

    x = torch.randn(5, 2)
    agent_idx = torch.tensor([0], dtype=torch.long)
    out = head(x, agent_idx, None)

    assert tuple(out.shape) == (16,)
    assert out.device == x.device


def test_sequential_head_agentic_encoding_changes_output_by_agent() -> None:
    torch.manual_seed(0)
    head = _make_sequential_head(agentic_encoding=True)

    x = torch.randn(2, 5, 2)
    x[1] = x[0]
    mask = torch.ones((2, 5), dtype=torch.bool)
    agent_idx = torch.tensor([[0], [1]], dtype=torch.long)

    out = head(x, agent_idx, mask)

    assert not torch.allclose(out[0], out[1], atol=1e-6, rtol=1e-6)


def test_sequential_head_sends_obs_plus_agent_embeddings_to_transformer() -> None:
    head = _make_sequential_head(agentic_encoding=True)
    embed_dim = head.config.embed_dim

    capture_transformer = _CaptureTransformer()
    head.obs_encoder = _ZeroObsEncoder(embed_dim)
    head.agent_embedder = _IndexAgentEmbedder(embed_dim)
    head.transformer = capture_transformer

    x = torch.randn(2, 4, 2)
    agent_idx = torch.tensor([[2], [1]], dtype=torch.long)

    _ = head(x, agent_idx, None)

    assert capture_transformer.last_input is not None
    expected = torch.tensor([2.0, 1.0], dtype=torch.float32).view(2, 1, 1).expand(2, 4, embed_dim)
    assert torch.allclose(capture_transformer.last_input, expected)


def test_sequential_head_masked_transformer_outputs_are_ignored_by_masked_mean() -> None:
    head = _make_sequential_head(agentic_encoding=False)
    embed_dim = head.config.embed_dim

    head.obs_encoder = _ZeroObsEncoder(embed_dim)
    head.transformer = _MaskedValueTransformer(valid_value=3.0, masked_value=1000.0)

    x = torch.randn(2, 5, 2)
    mask = torch.tensor(
        [[True, False, True, False, True], [False, True, True, True, False]],
        dtype=torch.bool,
    )
    agent_idx = torch.zeros((2, 1), dtype=torch.long)

    out = head(x, agent_idx, mask)

    assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-6, rtol=1e-6)