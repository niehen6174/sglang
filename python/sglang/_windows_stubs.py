"""Install compatibility stubs needed by SGLang Diffusion on native Windows.

Windows has no ``fcntl`` or ``resource`` module, and ``sgl_kernel`` publishes
no Windows build, yet all three are imported at module load across the shared
runtime. Rather than guarding each import site, ``sglang/__init__.py`` installs
small stand-ins once, as ``_platform_stubs.py`` does for Apple Silicon:

- ``fcntl``: ``flock`` and the ``LOCK_*`` flags, backed by LockFileEx /
  UnlockFileEx. The OS releases a lock when its handle closes or its process
  exits. Windows byte-range locks are mandatory rather than advisory.
- ``resource``: ``getrlimit`` reports unlimited and ``setrlimit`` does nothing,
  so the POSIX limit raising in the runtime is a no-op.
- ``sgl_kernel``: importing it and its submodules succeeds; calling any kernel
  raises. ``SGLANG_FORCE_FUSED_OP_BACKEND`` defaults to ``torch`` so fused ops
  run their reference instead of the CUDA forwards that call sgl_kernel.

Each stub is installed only when the real module is missing. This module
imports nothing from SGLang so it can run before the rest of the package.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types

# Submodules that must stay missing: their absence selects a fallback, e.g.
# the quantization utilities use MockScalarType when ``scalar_type`` is absent.
_SGL_KERNEL_MISSING_SUBMODULES = ("sgl_kernel.scalar_type",)

_installed = False


def _missing(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is None
    return importlib.util.find_spec(name) is None


class _UnavailableKernel:
    """An ``sgl_kernel`` attribute: importable, raises when called."""

    def __init__(self, name: str) -> None:
        self.__name__ = name
        self.__qualname__ = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__name__} is unavailable: sgl_kernel has no build here"
        )

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _UnavailableKernel(f"{self.__name__}.{name}")


class _SglKernelStub(types.ModuleType):
    """``sgl_kernel`` or one of its submodules."""

    is_stub = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []
        self.__spec__ = importlib.machinery.ModuleSpec(
            name, _SglKernelFinder, is_package=True
        )

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        if full in _SGL_KERNEL_MISSING_SUBMODULES:
            raise AttributeError(name)
        return _UnavailableKernel(full)


class _SglKernelFinder:
    """Resolves ``import sgl_kernel.<sub>`` to stub submodules."""

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if not fullname.startswith("sgl_kernel."):
            return None
        if fullname in _SGL_KERNEL_MISSING_SUBMODULES:
            return None
        return importlib.machinery.ModuleSpec(fullname, cls, is_package=True)

    @staticmethod
    def create_module(spec):
        return _SglKernelStub(spec.name)

    @staticmethod
    def exec_module(module):
        pass


def _install_sgl_kernel() -> None:
    sys.meta_path.append(_SglKernelFinder)
    sys.modules["sgl_kernel"] = _SglKernelStub("sgl_kernel")
    os.environ.setdefault("SGLANG_FORCE_FUSED_OP_BACKEND", "torch")


def _install_resource() -> None:
    resource = types.ModuleType("resource")
    resource.RLIM_INFINITY = sys.maxsize
    for i, name in enumerate(
        ("RLIMIT_NOFILE", "RLIMIT_STACK", "RLIMIT_MEMLOCK", "RLIMIT_AS")
    ):
        setattr(resource, name, i)
    resource.getrlimit = lambda _resource: (sys.maxsize, sys.maxsize)
    resource.setrlimit = lambda _resource, _limits: None
    sys.modules["resource"] = resource


def _install_fcntl() -> None:
    import ctypes
    import errno
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    lock_file = kernel32.LockFileEx
    lock_file.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    lock_file.restype = wintypes.BOOL
    unlock_file = kernel32.UnlockFileEx
    unlock_file.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    unlock_file.restype = wintypes.BOOL

    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
    LOCKFILE_FAIL_IMMEDIATELY, LOCKFILE_EXCLUSIVE_LOCK = 1, 2
    ERROR_LOCK_VIOLATION = 33

    def flock(fd, operation):
        if hasattr(fd, "fileno"):
            fd = fd.fileno()
        handle = msvcrt.get_osfhandle(fd)
        overlapped = _Overlapped()
        # Lock the whole file, including bytes past EOF.
        if operation == LOCK_UN:
            ok = unlock_file(
                handle, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlapped)
            )
        else:
            if (operation & ~LOCK_NB) not in (LOCK_SH, LOCK_EX):
                raise ValueError("Expected a shared, exclusive, or unlock operation")
            flags = (LOCKFILE_EXCLUSIVE_LOCK if operation & LOCK_EX else 0) | (
                LOCKFILE_FAIL_IMMEDIATELY if operation & LOCK_NB else 0
            )
            ok = lock_file(
                handle, flags, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlapped)
            )
        if not ok:
            error = ctypes.get_last_error()
            if error == ERROR_LOCK_VIOLATION:
                raise BlockingIOError(
                    errno.EAGAIN, "File lock is held by another process"
                )
            raise ctypes.WinError(error)

    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_SH, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN = (
        LOCK_SH,
        LOCK_EX,
        LOCK_NB,
        LOCK_UN,
    )
    fcntl.flock = flock
    sys.modules["fcntl"] = fcntl


def sgl_kernel_available() -> bool:
    """Whether the real sgl_kernel extension, not the stub, can be imported."""
    module = sys.modules.get("sgl_kernel")
    if module is not None:
        return not getattr(module, "is_stub", False)
    return not _missing("sgl_kernel")


def install_windows_stubs() -> None:
    """Install whichever of the stubs this interpreter needs. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    if _missing("sgl_kernel"):
        _install_sgl_kernel()
    if _missing("resource"):
        _install_resource()
    if sys.platform == "win32" and _missing("fcntl"):
        _install_fcntl()
