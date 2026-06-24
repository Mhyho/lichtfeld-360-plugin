# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Plugin for LichtFeld Studio."""

import glob
import os
import sys
from pathlib import Path


def _add_cuda_dll_dirs() -> None:
    """Register CUDA Toolkit + cuDNN bin directories on the DLL search path.

    The bundled GPU OpenCV (``opencv-contrib-python`` CUDA build) links against
    CUDA 13 runtime libraries (cudart/cublas/cufft/npp*) and cuDNN 9
    (``cudnn64_9.dll``). Python 3.8+ no longer resolves a native extension's
    dependencies via ``PATH``, so even with CUDA on the system PATH ``import
    cv2`` fails with "DLL load failed" unless these directories are registered
    explicitly. cuDNN ships in its own tree separate from the CUDA Toolkit.
    """

    def _add_best(patterns: list[str], required_dll: str) -> None:
        # Highest-versioned directory that actually contains the required DLL.
        for pattern in patterns:
            matches = sorted(
                (d for d in glob.glob(pattern)
                 if glob.glob(os.path.join(d, required_dll))),
                reverse=True,
            )
            if matches:
                best = matches[0]
                try:
                    os.add_dll_directory(best)
                except OSError:
                    pass
                # Belt-and-suspenders: also prepend to PATH. When Python is
                # embedded in a host .exe, add_dll_directory's user dirs are not
                # always honored for an extension's transitive deps, but the
                # loader still consults PATH for them.
                if best not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = best + os.pathsep + os.environ.get("PATH", "")
                return

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    cudnn_base = os.path.join(program_files, "NVIDIA", "CUDNN")

    # CUDA 13 toolkit runtime (cudart64_13.dll lives in bin\x64 on CUDA 13).
    _add_best(
        [
            os.path.join(cuda_base, "v13*", "bin", "x64"),
            os.path.join(cuda_base, "v13*", "bin"),
        ],
        "cudart64_13.dll",
    )
    # cuDNN 9 (separate install tree, e.g. CUDNN\v9.x\bin\<cuda>\x64).
    _add_best(
        [
            os.path.join(cudnn_base, "*", "bin", "*", "x64"),
            os.path.join(cudnn_base, "*", "bin", "*"),
            os.path.join(cudnn_base, "*", "bin"),
        ],
        "cudnn64_9.dll",
    )


if sys.platform == "win32":
    _lib_dir = Path(__file__).resolve().parent / "lib"
    if _lib_dir.is_dir():
        os.add_dll_directory(str(_lib_dir))
    _add_cuda_dll_dirs()
    # Pre-import cv2 once, after the CUDA/cuDNN DLL dirs are registered. This is
    # the correct point for the import to succeed; doing it here also prevents a
    # later failed import from leaving sys.OpenCV_LOADER set, which would mask
    # the real cause behind a spurious "recursion is detected" error on every
    # subsequent attempt in the same process.
    try:
        import cv2  # noqa: F401
    except BaseException:
        if hasattr(sys, "OpenCV_LOADER"):
            del sys.OpenCV_LOADER
        sys.modules.pop("cv2", None)

try:
    from .plugin import on_load, on_unload

    __all__ = ["on_load", "on_unload"]
except ImportError:
    # Allow importing sub-packages (e.g. core) outside the LichtFeld plugin
    # runtime — needed for standalone tests.
    pass
