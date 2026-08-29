"""One-shot child process for downloading the official SAM 3 weights."""

from __future__ import annotations

import argparse
import os
import sys

from squeakpose.services.model_download import download_sam3_weights
from squeakpose.workers.protocol import write_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    # The parent UI provides indeterminate progress; suppress multi-gigabyte tqdm
    # output so it is not retained in the process controller's diagnostics buffer.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        path = download_sam3_weights(args.destination)
    except Exception as exc:
        print(str(exc) or exc.__class__.__name__, file=sys.stderr, flush=True)
        return 1
    write_event({"event": "result", "model_path": path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
