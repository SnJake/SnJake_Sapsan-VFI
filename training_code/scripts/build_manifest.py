import argparse

from src.dataset import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--roots", type=str, nargs="+", required=True)
    parser.add_argument("--extensions", type=str, nargs="*", default=None)
    parser.add_argument("--backend", type=str, default="opencv")
    parser.add_argument("--errors", type=str, default=None)
    parser.add_argument("--force-count", action="store_true")
    args = parser.parse_args()

    count = build_manifest(
        args.roots,
        args.manifest,
        args.extensions,
        backend=args.backend,
        errors_path=args.errors,
        force_count=args.force_count,
    )
    print(f"Wrote {count} entries to {args.manifest}")


if __name__ == "__main__":
    main()
