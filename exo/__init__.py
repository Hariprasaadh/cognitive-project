"""CogExo: cognitive elbow exoskeleton for rehabilitation (simulation)."""
import os
import sys

# Anaconda ships an old MSVC runtime (14.29) next to python.exe, which gets loaded
# before the system one. Recent mujoco.dll builds need >= 14.4x and fail with
# "WinError 1114: DLL initialization routine failed". Preload the system runtime first.
if sys.platform == "win32":
    import ctypes

    _sys_rt = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
    if os.path.exists(_sys_rt):
        try:
            ctypes.WinDLL(_sys_rt)
        except OSError:
            pass
