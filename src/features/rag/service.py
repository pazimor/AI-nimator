# -*- coding: utf-8 -*-
"""
rag_openai.py — batch RAG/prompt runner using OpenAI API (Responses + Batch).

Covers:
- Single-file "test" run across one or many models.
- Create one or more Batch jobs from a generated JSONL (splits at ~180 MB).
- Retrieve all batch results and write prompts.json next to your original data.

CLI examples:
  # test one file
  python rag_openai.py test --file /path/to/x/anim/clip.vec.json

  # build & submit batch for everything under X with outputs written under Y
  python rag_openai.py batch \
      --in-dir /path/to/x \
      --out-root /path/to/y

  # fetch results, place prompts.json next to animation.json
  python rag_openai.py fetch --out-root /path/to/y

ENV:
  OPENAI_API_KEY   required
  OPENAI_BASE_URL  optional (for Azure OpenAI compatible endpoints)

Notes:
- Each batch JSONL line embeds a compact version of the .vec.json content (truncated/hashed if huge) and sets custom_id to the relative path of the vec file so we can put results next to the right animation.
"""
import json
import math
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# SDK v1
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class RagServiceError(RuntimeError):
    """Specific error raised by the RAG service when user input is invalid."""

from src.shared.constants import (
    RAG_BATCH_INPUTS_DIR,
    RAG_BATCH_OUTPUTS_DIR,
    RAG_JSONL_MAX_BYTES,
    RAG_ONE_LINE_SOFT_LIMIT,
    RAG_STATE_DIRNAME,
    RAG_STATE_FILE,
)
from src.shared.rag_utils import iter_prompt_entries_from_jsonl, write_prompt_payload
from src.shared.types import RagBatchJob, RagFetchJob, RagLocalFetchJob, RagTestJob


def load_state(root: Path) -> Dict[str, Any]:
    f = root / RAG_STATE_DIRNAME / RAG_STATE_FILE
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    # Only keep batch-related state; no file mapping
    return {"batches": {}}

def save_state(root: Path, state: Dict[str, Any]) -> None:
    sd = root / RAG_STATE_DIRNAME
    sd.mkdir(parents=True, exist_ok=True)
    (sd / RAG_STATE_FILE).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def compact_json_for_prompt(obj: Any, max_chars: Optional[int] = None) -> str:
    """
    Produce a compact single-line JSON (no spaces).
    If max_chars is None, do not truncate. If larger than max_chars, keep a head+tail window.
    """
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if max_chars is None or len(s) <= max_chars:
        return s
    marker = f'..."__TRUNCATED__":true,"__len__":{len(s)},"__sha1__":"{sha1_bytes(s.encode("utf-8"))}"...'
    keep = max_chars - len(marker)
    head = s[: keep // 2]
    tail = s[-(keep - len(head)) :]
    return head + marker + tail

def find_vec_files(in_dir: Path, glob_pat: str) -> List[Path]:
    return sorted([p for p in in_dir.glob(glob_pat) if p.suffix == ".json" and p.name.endswith(".vec.json")])

def openai_client():
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. pip install openai>=1.37.0")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # Force standard OpenAI unless user deliberately sets OPENAI_BASE_URL
    return OpenAI(base_url=base_url)

def ensure_dirs(out_root: Path):
    (out_root / RAG_STATE_DIRNAME).mkdir(parents=True, exist_ok=True)
    (out_root / RAG_BATCH_INPUTS_DIR).mkdir(parents=True, exist_ok=True)
    (out_root / RAG_BATCH_OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)



def _render_prompt(prompt_text: str, vec_payload: Any, rel_path: str) -> str:
    user_txt = (prompt_text or "").replace("{REL_PATH}", rel_path)
    user_txt = user_txt + "\n" + compact_json_for_prompt(vec_payload)
    return user_txt

def _extract_text_from_responses(resp_obj: Any) -> str:
    try:
        if hasattr(resp_obj, "output_text") and resp_obj.output_text:
            return resp_obj.output_text
    except Exception:
        pass
    return str(resp_obj)


def run_test(job: RagTestJob) -> str:
    in_file = job.file_path.expanduser().resolve()
    if not in_file.exists():
        raise RagServiceError(f"Missing file: {in_file}")

    vec = json.loads(in_file.read_text(encoding="utf-8"))
    rel_path = in_file.name
    client = openai_client()

    base_prompt = "Nom du fichier: {REL_PATH}\nContenu:"
    user_message = _render_prompt(base_prompt, vec, rel_path)

    try:
        response = client.responses.create(
            model=job.model,
            prompt={
                "id": "pmpt_68c5d759777c8197ae6b4c2e2ed5715c035330ad71bf7433",
                "version": "2",
            },
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_message}],
                }
            ],
            text={"format": "json_object"},
        )
    except Exception as exc:  # pragma: no cover - depends on network
        raise RagServiceError(f"OpenAI call failed: {exc}") from exc

    out_txt = _extract_text_from_responses(response)
    try:
        parsed = json.loads(out_txt)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return out_txt[:2000]

def build_jsonl_requests(
    in_dir: Path,
    out_root: Path,
    prompt_text: str,
    glob_pat: str,
    model: str,
    max_tokens_per_jsonl: Optional[int] = None,
    max_items_per_jsonl: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
    max_items: Optional[int] = None
) -> List[Path]:
    """
    Build one or more JSONL files under out_root/RAG_BATCH_INPUTS_DIR, splitting at ~180MB.
    Each line uses Responses API.
    """
    ensure_dirs(out_root)
    inputs_dir = out_root / RAG_BATCH_INPUTS_DIR
    inputs_dir.mkdir(parents=True, exist_ok=True)
    files = find_vec_files(in_dir, glob_pat)
    if not files:
        raise RagServiceError("No .vec.json found to batch.")
    part_idx = 1
    current_path = inputs_dir / f"batch_{int(time.time())}_{part_idx:03d}.jsonl"
    current_bytes = 0
    written = []
    fh = current_path.open("w", encoding="utf-8")
    current_tokens = 0
    total_tokens = 0
    items_count = 0
    current_lines = 0
    for p in (tqdm(files, desc="Preparing JSONL", unit="file") if tqdm else files):
        rel = p.relative_to(in_dir).as_posix()
        vec = json.loads(p.read_text(encoding="utf-8"))
        user_msg = _render_prompt(prompt_text, vec, rel)
        est_tokens = math.ceil(len(user_msg) / 4)
        if max_total_tokens is not None and total_tokens + est_tokens > max_total_tokens:
            break
        if max_items is not None and items_count >= max_items:
            break
        # If we've reached the per-JSONL item cap, roll over to a new file
        if max_items_per_jsonl is not None and current_lines >= max_items_per_jsonl:
            fh.close()
            if current_bytes > 0:
                written.append(current_path)
            part_idx += 1
            current_path = inputs_dir / f"batch_{int(time.time())}_{part_idx:03d}.jsonl"
            current_bytes = 0
            current_tokens = 0
            current_lines = 0
            fh = current_path.open("w", encoding="utf-8")
        # If adding this item would exceed the per-JSONL token cap, roll over to a new file
        if max_tokens_per_jsonl is not None and current_tokens + est_tokens > max_tokens_per_jsonl:
            fh.close()
            if current_bytes > 0:
                written.append(current_path)
            part_idx += 1
            current_path = inputs_dir / f"batch_{int(time.time())}_{part_idx:03d}.jsonl"
            current_bytes = 0
            current_tokens = 0
            current_lines = 0
            fh = current_path.open("w", encoding="utf-8")
        line = {
            "custom_id": rel,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "prompt": {
                    "id": "pmpt_68c5d759777c8197ae6b4c2e2ed5715c035330ad71bf7433",
                    "version": "2"
                },
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": user_msg}]}
                ],
            }
        }
        s = json.dumps(line, ensure_ascii=False) + "\n"
        b = s.encode("utf-8")
        if current_bytes + len(b) > RAG_JSONL_MAX_BYTES:
            fh.close()
            written.append(current_path)
            part_idx += 1
            current_path = inputs_dir / f"batch_{int(time.time())}_{part_idx:03d}.jsonl"
            current_bytes = 0
            current_tokens = 0
            current_lines = 0
            fh = current_path.open("w", encoding="utf-8")
        fh.write(s)
        current_bytes += len(b)
        current_tokens += est_tokens
        current_lines += 1
        total_tokens += est_tokens
        items_count += 1
    fh.close()
    if current_bytes > 0:
        written.append(current_path)
    return written

def submit_batch(client, input_file_id: str, completion_window: str = "24h") -> Dict[str, Any]:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        # output_file_id can be left empty; the service will create one upon completion
        # extra_body={"output_expires_after": {"seconds": 2592000, "anchor": "created_at"}},  # optional 30 days
    )
    return batch.to_dict() if hasattr(batch, "to_dict") else batch

def build_batches(job: RagBatchJob, dry_run: bool = False) -> Dict[str, Any]:
    ensure_dirs(job.output_root)
    state = load_state(job.output_root)
    client = openai_client()
    prompt_text = job.prompt_template
    jsonls = build_jsonl_requests(
        job.input_dir,
        job.output_root,
        prompt_text,
        job.glob_pattern,
        job.model,
        job.max_tokens_per_jsonl,
        job.max_items_per_jsonl,
        job.max_total_tokens,
        job.max_items,
    )
    if dry_run:
        total_chars = 0
        total_tokens = 0
        for jpath in jsonls:
            with jpath.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        body = row.get("body", {})
                        user_msg = body.get("input", [{}])[0].get("content", [{}])[0].get("text", "")
                        total_chars += len(user_msg)
                        total_tokens += len(user_msg)
                    except Exception:
                        continue
        return {
            "jsonl_files": [str(p) for p in jsonls],
            "total_chars": total_chars,
            "est_tokens_in": total_tokens,
        }

    created: List[Dict[str, Any]] = []
    for jpath in (tqdm(jsonls, desc="Uploading JSONL & Submitting Batches", unit="file") if tqdm else jsonls):
        with jpath.open("rb") as handle:
            up = client.files.create(file=handle, purpose="batch")
        batch_obj = submit_batch(client, up.id)
        created.append({"jsonl": jpath.name, "file_id": up.id, "batch_id": batch_obj["id"]})
        state.setdefault("batches", {})[batch_obj["id"]] = {
            "input_file_id": up.id,
            "jsonl": jpath.name,
            "status": batch_obj.get("status", "queued"),
            "output_file_id": None,
        }
    save_state(job.output_root, state)
    return {"submitted": created}


def fetch_batches(job: RagFetchJob) -> Dict[str, Any]:
    ensure_dirs(job.output_root)
    state = load_state(job.output_root)
    client = openai_client()

    for bid, meta in list(state.get("batches", {}).items()):
        try:
            batch_info = client.batches.retrieve(bid)
            bdict = batch_info.to_dict() if hasattr(batch_info, "to_dict") else batch_info
        except Exception as exc:
            print(f"[WARN] could not retrieve batch {bid}: {exc}", file=sys.stderr)
            continue
        state["batches"][bid]["status"] = bdict.get("status")
        if bdict.get("output_file_id"):
            state["batches"][bid]["output_file_id"] = bdict["output_file_id"]
    save_state(job.output_root, state)

    downloads = 0
    out_root = job.output_root
    for bid, meta in state.get("batches", {}).items():
        output_file_id = meta.get("output_file_id")
        if not output_file_id:
            continue
        try:
            body = client.files.content(output_file_id)
        except Exception as exc:
            print(f"[WARN] unable to download output for {bid}: {exc}", file=sys.stderr)
            continue
        downloads += 1
        text = body.read().decode("utf-8") if hasattr(body, "read") else str(body)
        for rel_vec_path, payload in iter_prompt_entries_from_jsonl(text.splitlines()):
            write_prompt_payload(out_root, rel_vec_path, payload)

    return {"downloads": downloads}


def process_local_batches(job: RagLocalFetchJob) -> Dict[str, Any]:
    """Populate `prompts.json` files using pre-downloaded JSONL responses.

    Parameters
    ----------
    job
        Execution parameters including the JSONL source directory and target
        animation root.

    Returns
    -------
    Dict[str, Any]
        Summary containing processed file count, number of prompts written and
        optionally the list of missing animations.
    """
    if not job.input_dir.exists():
        raise RagServiceError(f"Missing input directory: {job.input_dir}")
    if not job.target_root.exists():
        raise RagServiceError(f"Missing target directory: {job.target_root}")

    jsonl_files = sorted(p for p in job.input_dir.glob(job.glob_pattern) if p.is_file())
    if not jsonl_files:
        raise RagServiceError("No JSONL files found to process.")

    processed_files = 0
    prompts_written = 0
    missing_animation = set()

    exact_map, name_map = _build_animation_index(job.target_root)

    progress_iter = jsonl_files
    if tqdm:
        progress_iter = tqdm(jsonl_files, desc="Processing local JSONL", unit="file")

    for jsonl_path in progress_iter:
        processed_files += 1
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for relative_vec_path, payload in iter_prompt_entries_from_jsonl(handle):
                animation_folder = _resolve_animation_folder(
                    job.target_root,
                    relative_vec_path,
                    exact_map,
                    name_map,
                )
                if animation_folder is None:
                    missing_animation.add(str(relative_vec_path))
                    continue
                animation_folder.mkdir(parents=True, exist_ok=True)
                prompts_path = animation_folder / "prompts.json"
                if isinstance(payload, str):
                    content = payload
                else:
                    content = json.dumps(payload, indent=2, ensure_ascii=False)
                prompts_path.write_text(content, encoding="utf-8")
                prompts_written += 1

    result: Dict[str, Any] = {
        "processed_files": processed_files,
        "prompts_written": prompts_written,
    }
    if missing_animation:
        result["missing_animation"] = sorted(missing_animation)
    return result


def _build_animation_index(target_root: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    exact: Dict[str, Path] = {}
    name_map: Dict[str, List[Path]] = {}
    for anim_file in target_root.rglob("animation.json"):
        folder = anim_file.parent
        rel = folder.relative_to(target_root).as_posix()
        exact[rel] = folder

        candidates = {folder.name, rel}
        if "__" in folder.name:
            candidates.add(folder.name.split("__", 1)[0])
        if "__" in rel:
            candidates.add(rel.split("__", 1)[0])

        for key in candidates:
            name_map.setdefault(key, []).append(folder)
    return exact, name_map


def _resolve_animation_folder(
    target_root: Path,
    relative_vec_path: Path,
    exact_map: Dict[str, Path],
    name_map: Dict[str, List[Path]],
) -> Optional[Path]:
    """Locate the animation directory corresponding to a batch custom_id."""

    keys_to_try = []

    rel_parent = relative_vec_path.parent.as_posix()
    if rel_parent:
        keys_to_try.append(rel_parent)
        if "__" in rel_parent:
            keys_to_try.append(rel_parent.split("__", 1)[0])

    name = relative_vec_path.name
    if name.endswith(".vec.json"):
        base = name[: -len(".vec.json")]
    else:
        base = relative_vec_path.stem
    keys_to_try.append(base)
    if "__" in base:
        keys_to_try.append(base.split("__", 1)[0])

    for key in keys_to_try:
        if not key:
            continue
        if key in exact_map:
            return exact_map[key]
        matches = name_map.get(key)
        if matches:
            if len(matches) == 1:
                return matches[0]

    # Fallback: scan by directory name under target_root
    for key in keys_to_try:
        if not key:
            continue
        candidates = [p for p in target_root.glob(f"**/{key}") if (p / "animation.json").exists()]
        if len(candidates) == 1:
            return candidates[0]
        for cand in candidates:
            rel = cand.relative_to(target_root).as_posix()
            if rel.endswith(key):
                return cand
    return None


def cmd_test(args):
    job = RagTestJob(file_path=Path(args.file), model=args.model)
    preview = run_test(job)
    print(preview)
    return 0


def cmd_batch(args):
    job = RagBatchJob(
        input_dir=Path(args.in_dir),
        output_root=Path(args.out_root),
        prompt_template=getattr(args, "prompt", "Nom du fichier: {REL_PATH}\nContenu:"),
        glob_pattern=getattr(args, "glob", "**/*.vec.json"),
        model=args.model,
        max_tokens_per_jsonl=getattr(args, "max_tokens_per_jsonl", None),
        max_items_per_jsonl=getattr(args, "max_items_per_jsonl", None),
        max_total_tokens=getattr(args, "max_total_tokens", None),
        max_items=getattr(args, "max_items", None),
    )
    result = build_batches(job, dry_run=getattr(args, "dry_run", False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_fetch(args):
    job = RagFetchJob(output_root=Path(args.out_root))
    result = fetch_batches(job)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_fetch_local(args):
    job = RagLocalFetchJob(
        input_dir=Path(args.in_dir),
        target_root=Path(args.target_root),
        glob_pattern=getattr(args, "glob", "**/*.jsonl"),
    )
    result = process_local_batches(job)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "RagServiceError",
    "run_test",
    "build_batches",
    "fetch_batches",
    "process_local_batches",
    "cmd_test",
    "cmd_batch",
    "cmd_fetch",
    "cmd_fetch_local",
]
