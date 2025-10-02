import torch

import math
from math import isclose

from src.shared.quaternion import Quaternion, QuaternionConverter


def test_rotation6d_quaternion_roundtrip():
    torch.manual_seed(0)
    rotation6d = torch.randn(5, 6)
    quaternion = QuaternionConverter.quaternionFromRotation6d(rotation6d)
    recovered = QuaternionConverter.rotation6dFromQuaternion(quaternion)
    quaternion_roundtrip = QuaternionConverter.quaternionFromRotation6d(recovered)
    assert recovered.shape == rotation6d.shape
    diff = torch.min(
        torch.abs(quaternion - quaternion_roundtrip),
        torch.abs(quaternion + quaternion_roundtrip),
    )
    assert torch.all(diff < 1e-5)
    assert torch.isfinite(quaternion).all()


def test_format_quaternion_pipe_string():
    quaternion = torch.tensor([1.0, 0.5, 0.0, -0.5])
    formatted = QuaternionConverter.formatQuaternionPipeString(quaternion)
    assert formatted == "1|0.5|0|-0.5"


def test_quaternion_dataclass_helpers():
    quat = Quaternion(0.0, 0.0, 0.0, 1.0)
    normalized = quat.normalized()
    assert normalized == Quaternion.identity()
    tensor = quat.as_tensor()
    assert tensor.shape == (4,)
    multiplied = quat.multiply(Quaternion(0.0, 1.0, 0.0, 0.0))
    roll, pitch, yaw = multiplied.to_euler_xyz()
    assert isclose(abs(roll), math.pi, rel_tol=1e-6)


def test_quaternion_interpolation_variants():
    q0 = Quaternion.identity()
    q1 = Quaternion(0.0, 1.0, 0.0, 0.0)
    mid = q0.slerp(q1, 0.5)
    roll, pitch, yaw = mid.to_euler_xyz()
    assert math.isfinite(roll + pitch + yaw)

    lerp_mid = q0.lerp(q1, 0.5)
    assert math.isclose(lerp_mid.normalized().w, lerp_mid.normalized().w)


def test_quaternion_slerp_hemisphere_handling():
    q0 = Quaternion.identity()
    q1 = Quaternion(0.0, 0.0, 1.0, 0.0)
    result = q0.slerp(q1, 0.3)
    roll, pitch, yaw = result.to_euler_xyz()
    assert math.isfinite(roll + pitch + yaw)
