#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_device(preferred: str) -> str:
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_dtype(name: str, device: str):
    if device != "cuda":
        return None
    try:
        import torch
    except Exception:
        return None
    key = (name or "").strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _extract_caption(parsed: Any, task: str) -> Optional[str]:
    if isinstance(parsed, dict):
        for key in (task, "caption", "detailed_caption", "more_detailed_caption"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in parsed.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return None


def florence2_caption(
    *,
    image: Image.Image,
    model_id: str,
    task: str,
    max_new_tokens: int,
    device: str,
    dtype_name: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = _resolve_dtype(dtype_name, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model.to(device)
    model.eval()

    inputs = processor(text=task, images=image, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    prepared = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                prepared[key] = value.to(device=device, dtype=model_dtype)
            else:
                prepared[key] = value.to(device=device)
        else:
            prepared[key] = value
    inputs = prepared

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=task, image_size=image.size)
    caption = _extract_caption(parsed, task)
    return caption or generated_text


def promptgen_caption(
    *,
    image: Image.Image,
    model_id: str,
    max_new_tokens: int,
    device: str,
) -> str:
    from transformers import pipeline

    device_idx = 0 if device == "cuda" else -1
    pipe = pipeline("image-to-text", model=model_id, device=device_idx)
    result = pipe(image, generate_kwargs={"max_new_tokens": max_new_tokens})
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            for key in ("generated_text", "caption", "text"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(item, str) and item.strip():
            return item.strip()
    raise RuntimeError("promptgen returned no caption")


def main() -> int:
    parser = argparse.ArgumentParser(description="Image captioning helper.")
    parser.add_argument("--backend", required=True, choices=("florence2", "promptgen"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--task", default="<MORE_DETAILED_CAPTION>")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="float16")

    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    device = _resolve_device(args.device)

    try:
        if args.backend == "florence2":
            model_id = args.model or os.environ.get("FLORENCE2_MODEL") or "microsoft/Florence-2-base"
            caption = florence2_caption(
                image=image,
                model_id=model_id,
                task=args.task,
                max_new_tokens=int(args.max_new_tokens),
                device=device,
                dtype_name=args.dtype,
            )
        else:
            model_id = args.model or os.environ.get("PROMPTGEN_MODEL") or ""
            if not model_id:
                raise RuntimeError("promptgen backend requires --model or PROMPTGEN_MODEL")
            caption = promptgen_caption(
                image=image,
                model_id=model_id,
                max_new_tokens=int(args.max_new_tokens),
                device=device,
            )

        payload = {"caption": caption or ""}
        print(json.dumps(payload))
        return 0
    except Exception as exc:
        logging.exception("Captioning failed")
        payload = {"error": str(exc)}
        print(json.dumps(payload))
        return 1


if __name__ == "__main__":
    sys.exit(main())
