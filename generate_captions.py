#!/usr/bin/env python3
"""
Batch image description generation via OpenAI-compatible vision API.
Async concurrency + checkpoint resume.

Usage:
    python generate_captions.py sat          # satellite images
    python generate_captions.py svi_uav      # street-view / UAV images
"""

import argparse
import os
import json
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Literal
import aiohttp

API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_ENDPOINT = f"{API_BASE_URL}/chat/completions"

SCENE_CONFIG = {
    "sat":     {"prompt": "prompts/prompt_sat.txt",     "img_dir": "satellites"},
    "svi_uav": {"prompt": "prompts/prompt_svi_uav.txt", "img_dir": "svi_uav"},
}

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif",
    ".bmp": "image/bmp", ".webp": "image/webp",
}
IMAGE_EXTS = set(MIME_MAP.keys())


def get_api_key() -> str:
    key = os.getenv("API_KEY", "")
    if not key:
        raise RuntimeError("API_KEY env var is not set")
    return key


async def call_vision_api(
    session: aiohttp.ClientSession,
    image_path: Path,
    prompt: str,
    model: str,
    detail: Literal["low", "high"] = "high",
    temperature: float = 1.0,
    max_tokens: int = 300,
) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": detail,
                }},
            ],
        }],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with session.post(
        API_ENDPOINT, json=payload,
        headers={"Authorization": f"Bearer {get_api_key()}",
                 "Content-Type": "application/json"},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def process_batch(
    session: aiohttp.ClientSession,
    image_files: List[Path],
    prompt: str,
    output_file: Path,
    model: str,
    detail: Literal["low", "high"] = "high",
    temperature: float = 1.0,
    max_tokens: int = 300,
    batch_size: int = 15,
) -> List[Dict]:
    results: List[Dict] = []

    if output_file.exists():
        try:
            results = json.loads(output_file.read_text("utf-8"))
            print(f"Resumed {len(results)} from checkpoint.")
        except Exception:
            results = []

    done = {r["image_file"] for r in results if r.get("llm_response") is not None}
    remaining = [f for f in image_files if f.name not in done]

    if not remaining:
        print("Nothing to process.")
        return results

    print(f"Remaining: {len(remaining)}  |  Done: {len(done)}  |  Detail: {detail}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i : i + batch_size]
        print(f"\n[{i // batch_size + 1}/{n_batches}] {len(chunk)} images ...")

        coros = [call_vision_api(session, p, prompt, model, detail, temperature, max_tokens)
                 for p in chunk]
        outs = await asyncio.gather(*coros, return_exceptions=True)

        for img, out in zip(chunk, outs):
            entry = {"image_file": img.name, "image_path": str(img)}
            if isinstance(out, Exception):
                print(f"  ✗ {img.name}: {out}")
                entry.update(llm_response=None, error=str(out))
            else:
                print(f"  ✓ {img.name}")
                entry["llm_response"] = out
            results.append(entry)

        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
        print(f"  Saved checkpoint ({len(results)} total)")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Generate image descriptions via vision LLM")
    p.add_argument("scene", choices=SCENE_CONFIG.keys(), help="scene type")
    p.add_argument("--model", default="gpt-4o", help="model name (default: gpt-4o)")
    p.add_argument("--detail", default="high", choices=["low", "high"])
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


async def main():
    args = parse_args()
    cfg = SCENE_CONFIG[args.scene]
    root = Path(__file__).parent

    img_dir = root / cfg["img_dir"]
    prompt_file = root / cfg["prompt"]
    output_file = root / "output" / f"captions_{args.scene}.json"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")

    prompt = prompt_file.read_text("utf-8").strip()
    print(f"Scene: {args.scene}  |  Model: {args.model}  |  API: {API_ENDPOINT}")

    images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images in {img_dir}")
        return
    print(f"Found {len(images)} images.")

    async with aiohttp.ClientSession() as session:
        results = await process_batch(
            session, images, prompt, output_file,
            model=args.model, detail=args.detail,
            temperature=args.temperature, max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )

    ok = sum(1 for r in results if r.get("llm_response") is not None)
    print(f"\nDone: {ok} ok, {len(results) - ok} failed  ->  {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
