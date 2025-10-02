import json
from types import SimpleNamespace

from pathlib import Path

from src.features.rag import service
from src.shared.constants import RAG_STATE_DIRNAME, RAG_STATE_FILE
from src.shared.types import RagBatchJob, RagFetchJob, RagLocalFetchJob, RagTestJob


class DummyResponses:
    def __init__(self, text: str):
        self.text = text

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self.text)


class DummyFiles:
    def __init__(self, content: str = ""):
        self._content = content
        self.created = []

    def create(self, **kwargs):  # pragma: no cover - only used in non dry-run tests
        result = SimpleNamespace(id=f"file_{len(self.created)}")
        self.created.append(result)
        return result

    def content(self, file_id: str):
        return SimpleNamespace(read=lambda: self._content.encode("utf-8"))


class DummyBatches:
    def __init__(self, output_file_id: str):
        self._output_file_id = output_file_id
        self.created = []

    def create(self, input_file_id: str, endpoint: str, completion_window: str):
        batch_id = f"batch_{len(self.created)}"
        info = {"id": batch_id, "status": "queued", "output_file_id": None}
        self.created.append(info)
        return info

    def retrieve(self, batch_id: str):
        return {
            "id": batch_id,
            "status": "completed",
            "output_file_id": self._output_file_id,
        }


class DummyClient:
    def __init__(self, *, response_text: str = "{}", file_content: str = ""):
        self.responses = DummyResponses(response_text)
        self.files = DummyFiles(file_content)
        self.batches = DummyBatches("output_1")


def test_run_test(monkeypatch, tmp_path):
    vec = tmp_path / "clip.vec.json"
    vec.write_text(json.dumps({"sample": True}), encoding="utf-8")
    monkeypatch.setattr(service, "openai_client", lambda: DummyClient(response_text=json.dumps({"ok": True})))
    preview = service.run_test(RagTestJob(file_path=vec, model="dummy"))
    assert "ok" in preview


def test_build_batches_dry_run(monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    vec = in_dir / "clip.vec.json"
    vec.write_text(json.dumps({"sample": True}), encoding="utf-8")
    monkeypatch.setattr(service, "openai_client", lambda: DummyClient())
    job = RagBatchJob(
        input_dir=in_dir,
        output_root=tmp_path / "out",
        prompt_template="Nom du fichier: {REL_PATH}\nContenu:",
        glob_pattern="**/*.vec.json",
        model="dummy",
    )
    result = service.build_batches(job, dry_run=True)
    assert "jsonl_files" in result


def test_build_batches_submit(monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    out_root = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "clip.vec.json").write_text(json.dumps({"sample": True}), encoding="utf-8")
    dummy = DummyClient()
    monkeypatch.setattr(service, "openai_client", lambda: dummy)
    job = RagBatchJob(
        input_dir=in_dir,
        output_root=out_root,
        prompt_template="Nom du fichier: {REL_PATH}\nContenu:",
        glob_pattern="**/*.vec.json",
        model="dummy",
        max_items=1,
    )
    result = service.build_batches(job, dry_run=False)
    assert result["submitted"]
    state_file = out_root / RAG_STATE_DIRNAME / RAG_STATE_FILE
    assert state_file.exists()


def test_fetch_batches(monkeypatch, tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    state_dir = out_root / RAG_STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "batches": {
            "batch_1": {
                "input_file_id": "input",
                "jsonl": "file.jsonl",
                "status": "completed",
                "output_file_id": "output_1",
            }
        }
    }
    (state_dir / RAG_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    output_payload = json.dumps({
        "custom_id": "folder/clip.vec.json",
        "response": {"output": [{"content": [{"text": json.dumps({"prompt": "ok"})}]}]},
    })
    client = DummyClient(file_content=output_payload)
    monkeypatch.setattr(service, "openai_client", lambda: client)
    result = service.fetch_batches(RagFetchJob(output_root=out_root))
    assert result["downloads"] == 1
    prompts = next(out_root.rglob("prompts.json"))
    assert "prompt" in prompts.read_text(encoding="utf-8")


def test_process_local_batches(tmp_path):
    jsonl_dir = tmp_path / "batch_outputs"
    jsonl_dir.mkdir()
    target_root = tmp_path / "animations"
    nested = target_root / "folder"
    nested.mkdir(parents=True)
    (nested / "animation.json").write_text("{}", encoding="utf-8")

    payload = {
        "custom_id": "folder/clip.vec.json",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "{\n  \"prompt\": \"local\"\n}"}
                    ],
                }
            ],
        },
    }
    (jsonl_dir / "outputs.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    job = RagLocalFetchJob(input_dir=jsonl_dir, target_root=target_root)
    result = service.process_local_batches(job)

    assert result["processed_files"] == 1
    prompts_path = nested / "prompts.json"
    assert prompts_path.exists()
    assert "local" in prompts_path.read_text(encoding="utf-8")


def test_process_local_batches_missing_animation(tmp_path):
    jsonl_dir = tmp_path / "batch_outputs"
    jsonl_dir.mkdir()
    target_root = tmp_path / "animations"
    target_root.mkdir()

    payload = {
        "custom_id": "folder/clip.vec.json",
        "response": {"output": []},
    }
    (jsonl_dir / "outputs.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    job = RagLocalFetchJob(input_dir=jsonl_dir, target_root=target_root)
    result = service.process_local_batches(job)

    assert result["prompts_written"] == 0
    assert "missing_animation" in result
    assert result["missing_animation"] == ["folder/clip.vec.json"]


def test_process_local_batches_with_double_underscore(tmp_path):
    jsonl_dir = tmp_path / "batch_outputs"
    jsonl_dir.mkdir()
    target_root = tmp_path / "animations"
    animation_dir = target_root / "0000_motorcycle_poses_44c41da0"
    animation_dir.mkdir(parents=True)
    (animation_dir / "animation.json").write_text("{}", encoding="utf-8")

    payload = {
        "custom_id": "0000_motorcycle_poses_44c41da0__animation.vec.json",
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "{\n  \"Simple\": \"Le pilote ...\",\n  \"advanced\": \"...\",\n  \"tag\": \"Monture ou Véhicule\"\n}"
                            }
                        ],
                    }
                ]
            }
        },
    }
    (jsonl_dir / "outputs.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    job = RagLocalFetchJob(input_dir=jsonl_dir, target_root=target_root)
    result = service.process_local_batches(job)

    assert result["prompts_written"] == 1
    prompts_path = animation_dir / "prompts.json"
    assert prompts_path.exists()
    content = prompts_path.read_text(encoding="utf-8")
    assert content.startswith("{")
    assert "Monture ou Véhicule" in content
