"""A single linear layer demonstrating the model package layout."""

from torch import Tensor, nn


class DemoTorchModel(nn.Module):
    """Map float tensors of shape [batch, 4] to [batch, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)
