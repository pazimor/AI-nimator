"""Utility helpers shared across RAG features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple


def iter_prompt_entries_from_jsonl(lines: Iterable[str]) -> Iterator[Tuple[Path, str]]:
    """Yield prompt payloads decoded from JSONL batch output.

    Parameters
    ----------
    lines
        Iterable that yields lines from a JSONL file produced by the
        OpenAI batch Responses API.

    Yields
    ------
    Iterator[Tuple[Path, str]]
        Each tuple contains the relative path encoded in ``custom_id`` and the
        raw response text extracted from the batch output.
    """

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = _resolve_custom_id(row)
        if custom_id is None:
            continue
        payload_text = _extract_response_text(row)
        yield Path(custom_id), payload_text


def write_prompt_payload(base_output: Path, relative_vec_path: Path, payload: Any) -> Path:
    """Write the ``prompts.json`` file for a given animation folder.

    Parameters
    ----------
    base_output
        Root folder that mirrors the ``custom_id`` hierarchy.
    relative_vec_path
        Relative path pointing to the original ``*.vec.json`` input.
    payload
        Parsed prompt payload that will be dumped as JSON.

    Returns
    -------
    Path
        Absolute path to the ``prompts.json`` file that was written.
    """

    target_folder = base_output / relative_vec_path.parent
    target_folder.mkdir(parents=True, exist_ok=True)
    prompts_path = target_folder / "prompts.json"
    if isinstance(payload, str):
        content = payload
    else:
        content = json.dumps(payload, indent=2, ensure_ascii=False)
    prompts_path.write_text(content, encoding="utf-8")
    return prompts_path


def _resolve_custom_id(row: Dict[str, Any]) -> str | None:
    custom_id = row.get("custom_id")
    if isinstance(custom_id, str) and custom_id:
        return custom_id
    request = row.get("request")
    if isinstance(request, dict):
        inner_custom_id = request.get("custom_id")
        if isinstance(inner_custom_id, str) and inner_custom_id:
            return inner_custom_id
    return None


def _extract_response_text(row: Dict[str, Any]) -> str:
    response = row.get("response")
    if isinstance(response, dict):
        output_block = response.get("output")
        if isinstance(output_block, list):
            for item in output_block:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict) and "text" in chunk:
                            text_val = chunk.get("text")
                            if isinstance(text_val, str):
                                return text_val
                text_value = item.get("text")
                if isinstance(text_value, str):
                    return text_value
        body = response.get("body")
        if isinstance(body, dict):
            body_output = body.get("output")
            if isinstance(body_output, list):
                for item in body_output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                                return chunk["text"]
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        return text_value
            choices = body.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                                    return part["text"]
                    text_val = choice.get("text")
                    if isinstance(text_val, str):
                        return text_val
    if row.get("error") is not None:
        return json.dumps({"error": row["error"]}, ensure_ascii=False)
    return json.dumps(row, ensure_ascii=False)
