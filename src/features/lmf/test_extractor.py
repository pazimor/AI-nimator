import json
from pathlib import Path

import numpy as np

from src.features.lmf.extractor import extract_lmf
from src.features.lmf.bulk import run_bulk
from src.features.uniformizer import service as uniformizer_service
from src.shared.constants import LMF_JOINTS_SUBSET
from src.shared.types import UniformizerJob

SAMPLE_NPZ = Path(__file__).resolve().parents[3] / "tests" / "data" / "uniformizer" / "sample_source.npz"


def _prepare_sample_npz(destination: Path) -> Path:
    destination = Path(destination)
    if SAMPLE_NPZ.exists():
        destination.write_bytes(SAMPLE_NPZ.read_bytes())
    else:
        import numpy as np

        pos = np.zeros((2, 1, 3), dtype=np.float32)
        rot = np.zeros((2, 1, 4), dtype=np.float32)
        rot[..., -1] = 1.0
        np.savez_compressed(
            destination,
            positions=pos,
            rotations=rot,
            bone_names=np.array(["pelvis"], dtype=object),
            fps=30.0,
        )
    return destination


def test_extract_lmf(tmp_path):
    uniformized = tmp_path / "clip.json"
    source = _prepare_sample_npz(tmp_path / "source.npz")
    job = UniformizerJob(
        input_path=source,
        output_path=uniformized,
        target_skeleton="smpl22",
    )
    uniformizer_service.convert_npz_to_json(job)
    target = tmp_path / "clip.lmf.json"
    result = extract_lmf(uniformized, target, fps_internal=12.0)
    assert target.exists()
    assert result["meta"]["frames"] == result["meta"]["frames"]
    assert result["segments"]


def test_cli_bulk_converts_animation_json(tmp_path):
    def make_frame(offset: float) -> dict[str, object]:
        frame: dict[str, object] = {"root_pos": [0.0, offset, 0.0]}
        for joint in sorted(set(LMF_JOINTS_SUBSET) | {"pelvis"}):
            frame[joint] = [0.0, 0.0, 0.0, 1.0]
        return frame

    frames = [make_frame(i * 0.1) for i in range(4)]
    anim = {"meta": {"fps": 30.0}, "frames": frames}
    anim_path = tmp_path / "animation.json"
    anim_path.write_text(json.dumps(anim))

    logs: list[str] = []
    errors: list[str] = []
    exit_code = run_bulk(root=tmp_path, log=logs.append, err_log=errors.append)
    assert exit_code == 0
    assert not errors

    out_path = tmp_path / "animation.lmf.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["meta"]["frames"] == 2
