#!/usr/bin/env python3
"""
Check whether PyTorch can see a CUDA-capable GPU and show version details.
"""

import sys
import torch


def main() -> None:
    print(
        f"torch: {torch.__version__} | "
        f"compiled for CUDA {torch.version.cuda or 'CPU-only'}"
    )

    if not torch.cuda.is_available():
        # Exit with error message if no GPU is visible
        sys.exit("❌ PyTorch cannot see your GPU – install a CUDA build.")


if __name__ == "__main__":
    main()