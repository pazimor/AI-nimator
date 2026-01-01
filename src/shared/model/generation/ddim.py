import torch
import torch.nn as nn
import numpy as np

class DDIM(nn.Module):
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler.
    
    Handles the diffusion process mathematics: noise scheduling, alphas, betas.
    Does NOT contain the network architecture.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize DDIM.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of diffusion timesteps, by default 1000.
        beta_start : float, optional
            Starting value of beta, by default 0.0001.
        beta_end : float, optional
            Ending value of beta, by default 0.02.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Define beta schedule (linear)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Register as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Diffuse the data (add noise). q(x_t | x_0).

        Parameters
        ----------
        x_start : torch.Tensor
            Original data x_0.
        t : torch.Tensor
            Timesteps.
        noise : torch.Tensor, optional
            Noise tensor, by default None.

        Returns
        -------
        torch.Tensor
            Noisy data x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and noise.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy data at step t.
        t : torch.Tensor
            Timesteps.
        noise : torch.Tensor
            Predicted noise.

        Returns
        -------
        torch.Tensor
            Predicted x_0.
        """
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a at indices t and reshape to match x_shape.

        Parameters
        ----------
        a : torch.Tensor
            Source tensor.
        t : torch.Tensor
            Indices.
        x_shape : torch.Size
            Target shape.

        Returns
        -------
        torch.Tensor
            Extracted and reshaped tensor.
        """
        batch_size = t.shape[0]
        # Ensure both tensors are on the same device for gather
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
