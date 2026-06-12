import torch.nn as nn

class BaseLayer(nn.Module):
    """
    Base class for all layers in the model.
    
    Attributes
    ----------
    name : str
        The name of the layer.
    """

    def __init__(self, name: str = "base_layer"):
        """
        Initialize the BaseLayer.

        Parameters
        ----------
        name : str, optional
            The name of the layer, by default "base_layer".
        """
        super().__init__()
        self.name = name

    def forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")
