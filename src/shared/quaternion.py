"""Quaternion conversions, metrics, and core helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def normalizeQuaternionArray(quaternion: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a NumPy array (…×4) of unit quaternions."""

    arr = np.asarray(quaternion, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(np.isfinite(norms), norms, 0.0)
    norms = np.maximum(norms, eps)
    return arr / norms


def multiplyQuaternionArray(q0: np.ndarray, q1: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Multiply two NumPy quaternion arrays with normalization."""

    a = np.asarray(q0, dtype=np.float32)
    b = np.asarray(q1, dtype=np.float32)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    result = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return normalizeQuaternionArray(result, eps=eps)


def lerpQuaternionArray(q0: np.ndarray, q1: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Linearly interpolate two NumPy quaternions and normalize the result."""

    a = np.asarray(q0, dtype=np.float32)
    b = np.asarray(q1, dtype=np.float32)
    blended = (1.0 - float(alpha)) * a + float(alpha) * b
    return normalizeQuaternionArray(blended, eps=eps)


def slerpQuaternionArray(q0: np.ndarray, q1: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Spherical linear interpolation between two NumPy quaternions."""

    qa = normalizeQuaternionArray(q0, eps=eps)
    qb = normalizeQuaternionArray(q1, eps=eps)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return lerpQuaternionArray(qa, qb, alpha, eps=eps)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        return qa
    alpha = float(alpha)
    w0 = math.sin((1.0 - alpha) * theta) / sin_theta
    w1 = math.sin(alpha * theta) / sin_theta
    return w0 * qa + w1 * qb


@dataclass(frozen=True)
class Quaternion:
    """Simple quaternion helper bridging NumPy and Torch backends."""

    x: float
    y: float
    z: float
    w: float

    @staticmethod
    def from_iterable(values: Iterable[float]) -> "Quaternion":
        vx, vy, vz, vw = values
        return Quaternion(float(vx), float(vy), float(vz), float(vw))

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0.0, 0.0, 0.0, 1.0)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def as_numpy(self) -> np.ndarray:
        return np.array(self.as_tuple(), dtype=np.float32)

    def as_tensor(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        tensor = torch.tensor(self.as_tuple(), dtype=dtype or torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def normalized(self, eps: float = 1e-8) -> "Quaternion":
        arr = normalizeQuaternionArray(self.as_numpy(), eps=eps)
        return Quaternion.from_iterable(arr.tolist())

    def multiply(self, other: "Quaternion") -> "Quaternion":
        result = multiplyQuaternionArray(self.as_numpy(), other.as_numpy())
        return Quaternion.from_iterable(result.tolist())

    def to_euler_xyz(self) -> Tuple[float, float, float]:
        """Return Euler angles (radians) in XYZ order."""

        x, y, z, w = self.as_tuple()
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def lerp(self, other: "Quaternion", alpha: float) -> "Quaternion":
        result = lerpQuaternionArray(self.as_numpy(), other.as_numpy(), alpha)
        return Quaternion.from_iterable(result.tolist())

    def slerp(self, other: "Quaternion", alpha: float) -> "Quaternion":
        result = slerpQuaternionArray(self.as_numpy(), other.as_numpy(), alpha)
        return Quaternion.from_iterable(result.tolist())

class QuaternionConverter:
    """Conversions between quaternion formats used by the project."""

    @staticmethod
    def normalizeQuaternion(quaternion: Tensor) -> Tensor:
        """Return a unit-lengthed quaternion tensor."""
        epsilon = 1e-8
        norm = quaternion.norm(dim=-1, keepdim=True) + epsilon
        return quaternion / norm

    @staticmethod
    def quaternionFromRotation6d(rotation6d: Tensor) -> Tensor:
        """Convert 6D rotation representation into quaternions."""
        basisPrimary = F.normalize(rotation6d[..., 0:3], dim=-1)
        rawSecondary = rotation6d[..., 3:6]
        projection = (basisPrimary * rawSecondary).sum(dim=-1, keepdim=True)
        orthogonalComponent = rawSecondary - projection * basisPrimary
        basisSecondary = F.normalize(orthogonalComponent, dim=-1)
        basisTertiary = torch.cross(basisPrimary, basisSecondary, dim=-1)
        rotationMatrix = torch.stack(
            [basisPrimary, basisSecondary, basisTertiary], dim=-2
        )
        trace = rotationMatrix[..., 0, 0]
        trace = trace + rotationMatrix[..., 1, 1]
        trace = trace + rotationMatrix[..., 2, 2]
        wComponent = torch.sqrt(torch.clamp(1.0 + trace, min=0.0)) / 2.0
        denominator = 4.0 * wComponent + 1e-8
        xComponent = (
            rotationMatrix[..., 2, 1] - rotationMatrix[..., 1, 2]
        ) / denominator
        yComponent = (
            rotationMatrix[..., 0, 2] - rotationMatrix[..., 2, 0]
        ) / denominator
        zComponent = (
            rotationMatrix[..., 1, 0] - rotationMatrix[..., 0, 1]
        ) / denominator
        quaternion = torch.stack(
            [wComponent, xComponent, yComponent, zComponent], dim=-1
        )
        return QuaternionConverter.normalizeQuaternion(quaternion)

    @staticmethod
    def rotation6dFromQuaternion(quaternion: Tensor) -> Tensor:
        """Convert quaternions back to 6D rotation representation."""
        wComponent, xComponent, yComponent, zComponent = quaternion.unbind(-1)
        epsilon = 1e-8
        norm = torch.sqrt(
            wComponent * wComponent
            + xComponent * xComponent
            + yComponent * yComponent
            + zComponent * zComponent
            + epsilon
        )
        wComponent = wComponent / norm
        xComponent = xComponent / norm
        yComponent = yComponent / norm
        zComponent = zComponent / norm
        rotationMatrix = torch.zeros(
            quaternion.shape[:-1] + (3, 3),
            device=quaternion.device,
            dtype=quaternion.dtype,
        )
        rotationMatrix[..., 0, 0] = 1 - 2 * (
            yComponent * yComponent + zComponent * zComponent
        )
        rotationMatrix[..., 0, 1] = 2 * (
            xComponent * yComponent - zComponent * wComponent
        )
        rotationMatrix[..., 0, 2] = 2 * (
            xComponent * zComponent + yComponent * wComponent
        )
        rotationMatrix[..., 1, 0] = 2 * (
            xComponent * yComponent + zComponent * wComponent
        )
        rotationMatrix[..., 1, 1] = 1 - 2 * (
            xComponent * xComponent + zComponent * zComponent
        )
        rotationMatrix[..., 1, 2] = 2 * (
            yComponent * zComponent - xComponent * wComponent
        )
        rotationMatrix[..., 2, 0] = 2 * (
            xComponent * zComponent - yComponent * wComponent
        )
        rotationMatrix[..., 2, 1] = 2 * (
            yComponent * zComponent + xComponent * wComponent
        )
        rotationMatrix[..., 2, 2] = 1 - 2 * (
            xComponent * xComponent + yComponent * yComponent
        )
        primary = rotationMatrix[..., 0]
        secondary = rotationMatrix[..., 1]
        return torch.cat([primary, secondary], dim=-1)

    @staticmethod
    def quaternionFromAxisAngle(axisAngle: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Convert axis-angle vectors into quaternions.

        Parameters
        ----------
        axisAngle : Tensor
            Tensor shaped (..., 3) representing axis multiplied by angle.
        eps : float, default=1e-8
            Numerical stability guard near zero rotations.

        Returns
        -------
        Tensor
            Tensor shaped (..., 4) storing quaternions (w, x, y, z).
        """

        angle = torch.linalg.norm(axisAngle, dim=-1, keepdim=True)
        safeAngle = torch.where(angle > eps, angle, torch.ones_like(angle))
        normalizedAxis = axisAngle / safeAngle
        normalizedAxis = torch.where(
            angle > eps,
            normalizedAxis,
            torch.zeros_like(normalizedAxis),
        )
        halfAngle = angle * 0.5
        sinHalf = torch.sin(halfAngle)
        cosHalf = torch.cos(halfAngle)
        quaternion = torch.cat(
            [
                cosHalf,
                normalizedAxis * sinHalf,
            ],
            dim=-1,
        )
        return QuaternionConverter.normalizeQuaternion(quaternion)

    @staticmethod
    def rotation6dFromAxisAngle(axisAngle: Tensor) -> Tensor:
        """
        Convert axis-angle vectors directly into rotation-6d format.

        Parameters
        ----------
        axisAngle : Tensor
            Tensor shaped (..., 3).

        Returns
        -------
        Tensor
            Tensor shaped (..., 6) containing 6D rotations.
        """

        quaternions = QuaternionConverter.quaternionFromAxisAngle(axisAngle)
        return QuaternionConverter.rotation6dFromQuaternion(quaternions)

    @staticmethod
    def formatQuaternionPipeString(quaternion: Tensor) -> str:
        """Format a quaternion as the pipe-separated string used in JSON files."""
        formatted = [f"{float(component):.7f}" for component in quaternion]
        trimmed: List[str] = []
        for value in formatted:
            trimmed.append(value.rstrip("0").rstrip(".") if "." in value else value)
        return "|".join(trimmed)

    @staticmethod
    def tensorFromQuaternion(quaternion: Quaternion, device: torch.device | None = None) -> Tensor:
        return quaternion.as_tensor(device=device)

    @staticmethod
    def tensorSequenceFromQuaternions(quaternions: Iterable[Quaternion], device: torch.device | None = None) -> Tensor:
        data = [q.as_tuple() for q in quaternions]
        tensor = torch.tensor(data, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def quaternionFromTensor(tensor: Tensor) -> Quaternion:
        return Quaternion.from_iterable(tensor.tolist())


class QuaternionMetrics:
    """Metrics for quaternion quality estimates."""

    @staticmethod
    def geodesicDistanceDegrees(
        predicted: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute the geodesic distance between unit quaternions."""
        predicted = QuaternionConverter.normalizeQuaternion(predicted)
        target = QuaternionConverter.normalizeQuaternion(target)
        dotProduct = torch.clamp((predicted * target).sum(dim=-1).abs(), 0, 1)
        angleRadians = 2.0 * torch.acos(dotProduct)
        return angleRadians * (180.0 / math.pi)
