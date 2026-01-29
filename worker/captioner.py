#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _sanitize_enabled() -> bool:
    value = os.environ.get("CAPTION_SANITIZE", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _drop_patterns() -> list[re.Pattern]:
    defaults = [
        r"\b\d{1,2}\s*[- ]?year[- ]?old\b",
        r"\b(appears to be|seems to be)\b",
        r"\b(tattoo|tattoos|piercing|piercings)\b",
        r"\b(blue|green|brown|hazel|gray|grey)\s+eyes\b",
        r"\b(fair|pale|light|dark|tan)\s+skin\b",
        r"\b(overall mood|mood is|atmosphere)\b",
        r"\bslender\b",
        r"\bbreasts?\b",
    ]
    extra = os.environ.get("CAPTION_DROP_PATTERNS", "").strip()
    if extra:
        defaults.extend(p.strip() for p in extra.split(";") if p.strip())
    compiled: list[re.Pattern] = []
    for pattern in defaults:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _sanitize_caption(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<[^>]+>", "", text)
    text = " ".join(text.split())
    if not _sanitize_enabled():
        return text

    max_sentences = _env_int("CAPTION_MAX_SENTENCES", 2)
    patterns = _drop_patterns()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    seen = set()
    kept = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(p.search(sentence) for p in patterns):
            continue
        key = re.sub(r"[^a-z0-9]+", "", sentence.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
        if max_sentences and len(kept) >= max_sentences:
            break
    return " ".join(kept) or text


def _focus_enabled() -> bool:
    return _env_bool("CAPTION_FOCUS", True)


def _split_segments(text: str) -> list[str]:
    if not text:
        return []
    text = " ".join(text.split()).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        for part in re.split(r";\s*", piece):
            part = part.strip()
            if not part:
                continue
            if part.count(",") >= 2 and len(part.split()) > 12:
                for sub in re.split(r",\s*", part):
                    sub = sub.strip()
                    if sub:
                        segments.append(sub)
            else:
                segments.append(part)
    return segments


def _normalize_segment(segment: str) -> str:
    if not segment:
        return ""
    segment = re.sub(r"\s+", " ", segment).strip()
    return segment.strip(" .,:;!-")


def _parse_keywords(env_name: str, defaults: list[str]) -> list[str]:
    extra = os.environ.get(env_name, "").strip()
    keywords = list(defaults)
    if extra:
        extra_parts = re.split(r"[;,]\s*", extra)
        keywords.extend(k.strip() for k in extra_parts if k.strip())
    return keywords


def _score_segment(segment: str, keywords: list[str]) -> int:
    if not segment:
        return 0
    text = segment.lower()
    score = 0
    for kw in keywords:
        if kw and kw.lower() in text:
            score += 1
    return score


def _focus_caption(text: str) -> str:
    if not text or not _focus_enabled():
        return text

    segments = [_normalize_segment(s) for s in _split_segments(text)]
    segments = [s for s in segments if s]
    if not segments:
        return text

    head_keywords = _parse_keywords(
        "CAPTION_FOCUS_HEAD_KEYWORDS",
        [
            "face",
            "facial",
            "head",
            "jaw",
            "jawline",
            "chin",
            "cheek",
            "cheekbones",
            "nose",
            "nostril",
            "lips",
            "mouth",
            "teeth",
            "ears",
            "earring",
            "eyes",
            "iris",
            "pupil",
            "eyelid",
            "eyebrow",
            "lashes",
            "eyelashes",
            "forehead",
            "hair",
            "hairstyle",
            "bangs",
            "fringe",
            "ponytail",
            "braid",
            "bun",
            "skin",
            "complexion",
            "freckles",
            "moles",
            "beauty mark",
            "makeup",
            "lipstick",
            "eyeliner",
            "mascara",
        ],
    )
    body_keywords = _parse_keywords(
        "CAPTION_FOCUS_BODY_KEYWORDS",
        [
            "body",
            "figure",
            "physique",
            "posture",
            "pose",
            "stance",
            "torso",
            "waist",
            "hips",
            "legs",
            "arms",
            "hands",
            "fingers",
            "feet",
            "neck",
            "shoulders",
            "back",
            "chest",
            "bust",
            "breasts",
            "thighs",
            "calves",
            "dress",
            "skirt",
            "pants",
            "jeans",
            "shorts",
            "shirt",
            "blouse",
            "top",
            "jacket",
            "coat",
            "sweater",
            "hoodie",
            "lingerie",
            "underwear",
            "bra",
            "swimsuit",
            "bikini",
            "stockings",
            "gloves",
            "boots",
            "shoes",
            "heels",
            "necklace",
            "bracelet",
            "ring",
        ],
    )
    env_keywords = _parse_keywords(
        "CAPTION_FOCUS_ENV_KEYWORDS",
        [
            "background",
            "indoors",
            "outdoors",
            "room",
            "studio",
            "street",
            "city",
            "park",
            "beach",
            "forest",
            "mountain",
            "sky",
            "sunset",
            "sunrise",
            "lighting",
            "shadows",
            "window",
            "bed",
            "sofa",
            "chair",
            "wall",
            "floor",
            "scene",
            "setting",
        ],
    )

    head: list[str] = []
    body: list[str] = []
    env: list[str] = []
    other: list[str] = []
    for segment in segments:
        head_score = _score_segment(segment, head_keywords)
        body_score = _score_segment(segment, body_keywords)
        env_score = _score_segment(segment, env_keywords)
        if head_score == 0 and body_score == 0 and env_score == 0:
            other.append(segment)
        elif head_score >= body_score and head_score >= env_score:
            head.append(segment)
        elif body_score >= env_score:
            body.append(segment)
        else:
            env.append(segment)

    max_segments = _env_int("CAPTION_FOCUS_MAX_SEGMENTS", 6)
    if max_segments <= 0 or max_segments >= len(segments):
        focused = head + body + env + other
        return ". ".join(focused) or text

    head_weight = _env_float("CAPTION_FOCUS_HEAD_WEIGHT", 0.92)
    body_weight = _env_float("CAPTION_FOCUS_BODY_WEIGHT", 0.08)
    env_weight = _env_float("CAPTION_FOCUS_ENV_WEIGHT", 0.0)
    other_weight = max(0.0, 1.0 - head_weight - body_weight - env_weight)
    weights = [
        ("head", head, head_weight),
        ("body", body, body_weight),
        ("env", env, env_weight),
        ("other", other, other_weight),
    ]

    head_priority = _env_bool("CAPTION_FOCUS_HEAD_PRIORITY", True)
    if head_priority and head:
        selected = head[:max_segments]
        remaining = max_segments - len(selected)
        if remaining <= 0:
            return ". ".join(selected) or text

        groups = [
            ("body", body, body_weight),
            ("env", env, env_weight),
            ("other", other, other_weight),
        ]
        total_weight = sum(weight for _name, _items, weight in groups if weight > 0)
        targets: dict[str, int] = {}
        if total_weight > 0:
            for name, items, weight in groups:
                targets[name] = min(len(items), int(round(remaining * (weight / total_weight))))
        else:
            for name, _items, _weight in groups:
                targets[name] = 0

        total = sum(targets.values())
        if total < remaining:
            for name, items, weight in sorted(groups, key=lambda item: item[2], reverse=True):
                while total < remaining and targets[name] < len(items):
                    targets[name] += 1
                    total += 1
        elif total > remaining:
            for name, _items, weight in sorted(groups, key=lambda item: item[2]):
                while total > remaining and targets[name] > 0:
                    targets[name] -= 1
                    total -= 1

        for name, items, _weight in groups:
            take = targets.get(name, 0)
            if take > 0:
                selected.extend(items[:take])

        return ". ".join(selected) or text

    targets: dict[str, int] = {}
    for name, items, weight in weights:
        targets[name] = int(round(max_segments * max(0.0, weight)))

    if head and targets["head"] == 0:
        targets["head"] = 1

    total = sum(targets.values())
    if total > max_segments:
        for name in ("other", "env", "body", "head"):
            while total > max_segments and targets[name] > (1 if name == "head" and head else 0):
                targets[name] -= 1
                total -= 1
    elif total < max_segments:
        for name, items, _weight in sorted(weights, key=lambda item: item[2], reverse=True):
            while total < max_segments and targets[name] < len(items):
                targets[name] += 1
                total += 1

    selected: list[str] = []
    for name, items, _weight in weights:
        take = targets.get(name, 0)
        if take > 0:
            selected.extend(items[:take])

    return ". ".join(selected) or text


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
    output = caption or generated_text
    output = _sanitize_caption(output) or output
    return _focus_caption(output)


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
                    output = _sanitize_caption(value.strip()) or value.strip()
                    return _focus_caption(output)
        if isinstance(item, str) and item.strip():
            output = _sanitize_caption(item.strip()) or item.strip()
            return _focus_caption(output)
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
