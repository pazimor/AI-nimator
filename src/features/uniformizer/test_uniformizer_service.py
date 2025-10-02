import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.features.uniformizer import service
from src.shared.types import UniformizerDirectoryJob, UniformizerJob

SAMPLE_NPZ = Path(__file__).resolve().parents[3] / "tests" / "data" / "uniformizer" / "sample_source.npz"


def _prepare_sample_npz(destination: Path) -> Path:
    destination = Path(destination)
    if SAMPLE_NPZ.exists():
        destination.write_bytes(SAMPLE_NPZ.read_bytes())
    else:
        _make_npz(destination)
    return destination


def _make_npz(path: Path) -> None:
    pos = np.zeros((2, 1, 3), dtype=np.float32)
    rot = np.zeros((2, 1, 4), dtype=np.float32)
    rot[..., -1] = 1.0
    np.savez_compressed(
        path,
        positions=pos,
        rotations=rot,
        bone_names=np.array(["pelvis"], dtype=object),
        fps=30.0,
    )


def test_convert_npz_to_json(tmp_path):
    source = _prepare_sample_npz(tmp_path / "sample_source.npz")
    dst = tmp_path / "uniformized.json"
    job = UniformizerJob(
        input_path=source,
        output_path=dst,
        target_skeleton="smpl22",
        include_prompts=True,
    )
    frames, fps = service.convert_npz_to_json(job)
    assert frames > 0
    assert fps > 0
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["meta"]["format"] == "rotroot"
    assert (dst.parent / "prompts.json").exists()


def test_convert_directory(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    _prepare_sample_npz(src_dir / "a.npz")
    job = UniformizerDirectoryJob(
        input_dir=src_dir,
        output_dir=out_dir,
        include_prompts=True,
    )
    processed, skipped = service.convert_directory(job)
    assert processed == 1
    assert skipped == 0
    assert any(p.name == "prompts.json" for p in out_dir.rglob("prompts.json"))


def test_command_wrappers(tmp_path):
    src = _prepare_sample_npz(tmp_path / "sample.npz")
    dst = tmp_path / "sample.json"
    args = SimpleNamespace(
        npz=str(src),
        json=str(dst),
        target_skel=None,
        target_map=None,
        resample=None,
        prompts=True,
    )
    assert service.cmd_npz2json(args) == 0
    assert (dst.parent / "prompts.json").exists()
    args2 = SimpleNamespace(json=str(dst), npz=str(tmp_path / "back.npz"))
    assert service.cmd_json2npz(args2) == 0
