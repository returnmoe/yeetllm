from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="internal static LoRA downloader")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    downloaded = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(downloaded)


if __name__ == "__main__":
    main()
