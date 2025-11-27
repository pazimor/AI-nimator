from typing import Union

import kornia.geometry.conversions as K
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


###############################
# Unified rotation class (Torch backend)
###############################

class Rotation:
    """
    Unified rotation representation backed by quaternions (x, y, z, w).

    You can construct it from:
      - Euler angles (shape (..., 3))  -> radians, order: xyz
      - Quaternions (shape (..., 4))   -> (x, y, z, w)
      - 6D rotation (shape (..., 6))   -> first 2 columns of 3x3 matrix

    Internal: everything is stored as normalized quaternion (Torch Tensor).
    """

    def __init__(self, data: Union[np.ndarray, Tensor], kind: str = "quat") -> None:
        kind = kind.lower()
        t = self._to_tensor(data)

        if kind == "quat":
            quat = self._ensure_last_dim(t, 4)
            # Kornia expects (w, x, y, z) for some ops, but let's stick to our internal storage (x, y, z, w).
            # We will normalize it.
            quat = self._normalize_quat(quat)

        elif kind == "euler":
            # angles in radians, assumed xyz order
            angles = self._ensure_last_dim(t, 3)
            # Kornia quaternion_from_euler expects (roll, pitch, yaw) -> (x, y, z)
            # Returns tuple (w, x, y, z)
            w, x, y, z = K.quaternion_from_euler(angles[..., 0], angles[..., 1], angles[..., 2])
            # Stack to (x, y, z, w)
            quat = torch.stack((x, y, z, w), dim=-1)

        elif kind in ("rot6d", "6d"):
            r6 = self._ensure_last_dim(t, 6)
            # Manual 6D -> Matrix -> Quaternion
            mat = self._rotation_6d_to_matrix(r6)
            # Kornia rotation_matrix_to_quaternion expects (..., 3, 3)
            # Returns (..., 4) in (w, x, y, z) order
            quat_wxyz = K.rotation_matrix_to_quaternion(mat)
            # Convert (w, x, y, z) -> (x, y, z, w)
            quat = torch.cat((quat_wxyz[..., 1:], quat_wxyz[..., :1]), dim=-1)

        else:
            raise ValueError(
                f"Rotation.__init__: kind must be 'quat', 'euler' or 'rot6d', got {kind!r}"
            )

        self._quat: Tensor = quat  # (..., 4), float32, (x, y, z, w)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def quat(self) -> Tensor:
        """Internal quaternion, shape (..., 4), (x, y, z, w)."""
        return self._quat

    @property
    def euler(self) -> Tensor:
        """
        Euler angles (xyz) in radians, shape (..., 3).
        """
        # Convert (x, y, z, w) -> (w, x, y, z) for Kornia
        w = self._quat[..., 3]
        x = self._quat[..., 0]
        y = self._quat[..., 1]
        z = self._quat[..., 2]
        
        # Kornia euler_from_quaternion takes (w, x, y, z)
        # Returns tuple (roll, pitch, yaw) -> (x, y, z)
        roll, pitch, yaw = K.euler_from_quaternion(w, x, y, z)
        return torch.stack((roll, pitch, yaw), dim=-1)

    @property
    def rot6d(self) -> Tensor:
        """
        6D representation, shape (..., 6).
        """
        # Convert (x, y, z, w) -> (w, x, y, z) for Kornia
        quat_wxyz = torch.cat((self._quat[..., -1:], self._quat[..., :-1]), dim=-1)
        mat = K.quaternion_to_rotation_matrix(quat_wxyz)  # (..., 3, 3)
        return self._rotation_matrix_to_rotation_6d(mat)

    @property
    def axis_angle(self) -> Tensor:
        """
        Axis-angle representation, shape (..., 3).
        """
        # Convert (x, y, z, w) -> (w, x, y, z) for Kornia
        quat_wxyz = torch.cat((self._quat[..., -1:], self._quat[..., :-1]), dim=-1)
        return K.quaternion_to_axis_angle(quat_wxyz)

    # ------------------------------------------------------------------
    # Generic API
    # ------------------------------------------------------------------

    def as_tensor(self, kind: str = "quat") -> Tensor:
        """Returns the requested representation as a Torch Tensor.

        kind ∈ {"quat", "euler", "rot6d", "6d"}
        """
        kind = kind.lower()
        if kind == "quat":
            return self.quat
        if kind == "euler":
            return self.euler
        if kind in ("rot6d", "6d"):
            return self.rot6d
        raise ValueError(f"Rotation.as_tensor: unknown kind {kind!r}")

    def as_array(self, kind: str = "quat") -> np.ndarray:
        """Same as as_tensor but converted to NumPy on CPU."""
        return self.as_tensor(kind).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Constructors helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_quat(cls, quat: Union[np.ndarray, Tensor]) -> "Rotation":
        return cls(quat, kind="quat")

    @classmethod
    def from_euler(cls, angles: Union[np.ndarray, Tensor]) -> "Rotation":
        """Angles in radians, xyz order."""
        return cls(angles, kind="euler")

    @classmethod
    def from_rot6d(cls, r6: Union[np.ndarray, Tensor]) -> "Rotation":
        return cls(r6, kind="rot6d")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32)

    @staticmethod
    def _ensure_last_dim(t: Tensor, dim: int) -> Tensor:
        if t.ndim == 1 and t.shape[0] == dim:
            t = t.unsqueeze(0)
        if t.shape[-1] != dim:
            raise ValueError(
                f"Rotation: last dimension must be {dim}, got {tuple(t.shape)}"
            )
        return t

    @staticmethod
    def _normalize_quat(quat: Tensor) -> Tensor:
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return quat / norm

    # ------------------------------------------------------------------
    # Manual 6D Helpers (Missing in Kornia 0.x)
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_6d_to_matrix(d6: Tensor) -> Tensor:
        """
        Converts 6D rotation representation to 3x3 rotation matrix.
        Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".
        """
        a1 = d6[..., 0:3]
        a2 = d6[..., 3:6]

        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        return torch.stack((b1, b2, b3), dim=-1)

    @staticmethod
    def _rotation_matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
        """
        Converts 3x3 rotation matrix to 6D representation.
        """
        batch_dim = matrix.shape[:-2]
        return matrix[..., :2].clone().reshape(batch_dim + (6,))