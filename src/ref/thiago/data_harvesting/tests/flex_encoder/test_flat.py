import torch

from data_harvesting.encoder import FlatEncoder, FlatEncoderConfig, FlatEncoderInput


def _make_flat_head() -> FlatEncoder:
    return FlatEncoder(
        input=FlatEncoderInput(key="agent_id", input_size=2),
        config=FlatEncoderConfig(
            embed_dim=8,
            depth=1,
            num_cells=16,
            activation_class=torch.nn.Tanh,
        ),
        device=torch.device("cpu"),
    )


def test_flat_head_output_shape_preserves_single_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_flat_head()

    x = torch.randn(7, 2)
    out = head(x)

    assert tuple(out.shape) == (7, 8)
    assert out.device == x.device


def test_flat_head_output_shape_preserves_multi_batch_dims() -> None:
    torch.manual_seed(0)
    head = _make_flat_head()

    x = torch.randn(2, 3, 2)
    out = head(x)

    assert tuple(out.shape) == (2, 3, 8)
    assert out.device == x.device


def test_flat_head_output_shape_preserves_no_batch_dim() -> None:
    torch.manual_seed(0)
    head = _make_flat_head()

    x = torch.randn(2)
    out = head(x)

    assert tuple(out.shape) == (8,)
    assert out.device == x.device
