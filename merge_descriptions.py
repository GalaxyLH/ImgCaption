#!/usr/bin/env python3
"""
Merge LLM-generated descriptions into existing pickle datasets.
Result: [env desc] + "Specifically, the target is to " + [find ...] + [area/distance]

Usage:
    python merge_descriptions.py                           # default: output/captions.json
    python merge_descriptions.py --captions output/captions_sat.json
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple

SKIP_PHRASES = [
    "i'm sorry", "i can't", "i'm unable",
    "can't assist", "can't analyze", "can't help", "cannot",
]


def load_llm_responses(json_path: Path) -> Dict[str, str]:
    """Load {filename: text} from captions JSON, drop refusals."""
    print(f"Loading from {json_path} ...")
    raw = json.loads(json_path.read_text("utf-8"))

    out: Dict[str, str] = {}
    for item in raw:
        name = item.get("image_file") or ""
        if not name:
            p = item.get("image_path", "")
            name = Path(p).name if p else ""

        text = item.get("llm_response") or item.get("caption") or ""
        if not name or not text:
            continue
        if any(p in text.lower() for p in SKIP_PHRASES):
            continue
        out[name] = text

    print(f"  Valid: {len(out)}")
    return out


def _split_find(desc: str) -> Tuple[str, str]:
    desc = desc.strip()
    for kw in ("The area", "The distance"):
        if kw in desc:
            before, after = desc.split(kw, 1)
            return before.strip().rstrip("."), kw + after
    return desc.rstrip("."), ""


def merge_text(old_desc: str, env_desc: str,
               connector: str = "Specifically, the target is to ") -> str:
    find_part, supplement = _split_find(old_desc)
    if find_part.startswith("Find "):
        find_part = "find " + find_part[5:]

    merged = f"{env_desc.strip()} {connector}{find_part}"
    if supplement:
        merged += f". {supplement.strip()}"
    return merged


def update_pickle(pkl_path: Path, responses: Dict[str, str],
                  out_dir: Path) -> Tuple[int, int]:
    print(f"\n  {pkl_path.name} ...")
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    print(f"    {len(records)} records")

    updated = missing = 0
    for i, rec in enumerate(records):
        if len(rec) < 8:
            continue
        sat = rec[0]
        if sat not in responses:
            missing += 1
            if missing <= 5:
                print(f"    Missing: {sat}")
            continue

        merged = merge_text(rec[2], responses[sat])
        records[i] = (*rec[:2], merged, *rec[3:])
        updated += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / pkl_path.name
    with open(out, "wb") as f:
        pickle.dump(records, f)
    print(f"    Updated {updated}, missing {missing}  ->  {out}")
    return updated, missing


def main():
    parser = argparse.ArgumentParser(description="Merge LLM descriptions into pickle datasets")
    parser.add_argument("--captions", default="output/captions.json",
                        help="path to captions JSON (relative to script dir)")
    args = parser.parse_args()

    base = Path(__file__).parent
    captions_json = base / args.captions
    old_dir = base / "text" / "old" / "sRSVG"
    new_dir = base / "text" / "new" / "sRSVG"

    if not captions_json.exists():
        raise FileNotFoundError(captions_json)
    if not old_dir.exists():
        raise FileNotFoundError(old_dir)

    responses = load_llm_responses(captions_json)

    total_up = total_miss = 0
    for name in ("sRSVG_train.pickle", "sRSVG_val.pickle", "sRSVG_test.pickle"):
        pkl = old_dir / name
        if not pkl.exists():
            print(f"  Skipped: {name}")
            continue
        u, m = update_pickle(pkl, responses, new_dir)
        total_up += u
        total_miss += m

    total = total_up + total_miss
    rate = f"{total_up / total * 100:.1f}%" if total else "N/A"
    print(f"\nUpdated {total_up}, missing {total_miss} ({rate})  ->  {new_dir}")


if __name__ == "__main__":
    main()
