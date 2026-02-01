import argparse
import os
import random
import shutil
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from src.dataset import scan_videos, build_manifest


def sanitize(name: str) -> str:
    bad = [":", "\\", "/", "*", "?", "\"", "<", ">", "|"]
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def parse_root_limits(root_limits: Sequence[str]) -> List[Tuple[str, int]]:
    pairs = []
    for item in root_limits:
        if "=" not in item:
            raise ValueError(f"Invalid --root-limit '{item}', expected PATH=NUM")
        path, num = item.rsplit("=", 1)
        pairs.append((path, int(num)))
    return pairs


def choose_files(files: List[str], limit: int, seed: int) -> List[str]:
    if limit <= 0 or limit >= len(files):
        return files
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    return files[:limit]


def copy_mode(src: str, dst: str, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    raise ValueError(f"Unknown mode: {mode}")


def prepare_dataset(
    out_dir: str,
    root_limits: List[Tuple[str, int]],
    extensions: Sequence[str],
    seed: int,
    mode: str,
    skip_existing: bool,
    max_total: int,
    dry_run: bool,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    selected_all: List[str] = []
    total_count = 0
    for idx, (root, limit) in enumerate(root_limits):
        root = os.path.normpath(root)
        files = scan_videos([root], extensions)
        chosen = choose_files(files, limit, seed + idx)
        if max_total > 0:
            remaining = max_total - total_count
            if remaining <= 0:
                break
            chosen = chosen[:remaining]
        root_name = f"root_{idx+1:02d}_{sanitize(os.path.basename(root))}"
        for src in chosen:
            rel = os.path.relpath(src, root)
            dst = os.path.join(out_dir, root_name, rel)
            selected_all.append((src, dst))
        total_count += len(chosen)
    to_copy = selected_all
    if dry_run:
        return [dst for _, dst in to_copy]

    pbar = tqdm(to_copy, desc="copy", total=len(to_copy))
    copied = []
    for src, dst in pbar:
        if skip_existing and os.path.exists(dst):
            copied.append(dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            copy_mode(src, dst, mode)
            copied.append(dst)
        except Exception as exc:
            pbar.set_postfix({"err": str(exc)[:80]})
            continue
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--root-limit", type=str, nargs="*", default=[])
    parser.add_argument("--root", type=str, nargs="*", default=[])
    parser.add_argument("--limit", type=int, nargs="*", default=[])
    parser.add_argument("--extensions", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "hardlink", "symlink"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", type=str, default=None)
    parser.add_argument("--manifest-backend", type=str, default="opencv")
    args = parser.parse_args()

    if args.root_limit:
        root_limits = parse_root_limits(args.root_limit)
    else:
        if not args.root:
            raise ValueError("Specify --root-limit or --root")
        if args.limit and len(args.limit) != len(args.root):
            raise ValueError("--limit count must match --root count")
        limits = args.limit or [0 for _ in args.root]
        root_limits = list(zip(args.root, limits))

    extensions = args.extensions or [".mp4", ".mkv", ".avi", ".webm", ".flv", ".mov", ".mpg", ".mpeg"]
    copied = prepare_dataset(
        out_dir=args.out,
        root_limits=root_limits,
        extensions=extensions,
        seed=args.seed,
        mode=args.mode,
        skip_existing=args.skip_existing,
        max_total=int(args.max_total),
        dry_run=args.dry_run,
    )

    print(f"Prepared {len(copied)} files in {args.out}")
    if args.write_manifest and not args.dry_run:
        build_manifest([args.out], args.write_manifest, extensions, backend=args.manifest_backend)
        print(f"Wrote manifest to {args.write_manifest}")


if __name__ == "__main__":
    main()
