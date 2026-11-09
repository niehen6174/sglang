"""Descriptor-based shared/exclusive file locks on POSIX and Windows.

Windows locks the entire file, including bytes beyond EOF, using LockFileEx.
The OS releases the lock when the descriptor closes or its process exits.
"""

import sys

if sys.platform != "win32":
    from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock
else:
    import ctypes
    import errno
    import msvcrt
    from ctypes import wintypes

    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _lock = _kernel32.LockFileEx
    _lock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _lock.restype = wintypes.BOOL
    _unlock = _kernel32.UnlockFileEx
    _unlock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _unlock.restype = wintypes.BOOL

    def flock(fd, operation):
        if hasattr(fd, "fileno"):
            fd = fd.fileno()
        handle = msvcrt.get_osfhandle(fd)
        overlapped = _Overlapped()
        if operation == LOCK_UN:
            success = _unlock(
                handle, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlapped)
            )
        else:
            if (operation & ~LOCK_NB) not in (LOCK_SH, LOCK_EX):
                raise ValueError("Expected a shared, exclusive, or unlock operation")
            flags = (2 if operation & LOCK_EX else 0) | (
                1 if operation & LOCK_NB else 0
            )
            success = _lock(
                handle, flags, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlapped)
            )
        if not success:
            error = ctypes.get_last_error()
            if error == 33:  # ERROR_LOCK_VIOLATION
                raise BlockingIOError(
                    errno.EAGAIN, "File lock is held by another process"
                )
            raise ctypes.WinError(error)
