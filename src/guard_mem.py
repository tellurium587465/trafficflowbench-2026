"""Admission control for heavy jobs: exit 1 (and say why) if free physical memory is below the requirement.
The PC hard-reset on 10/7 12:01 coincided with free memory hitting 4 MB (LGBM final ~14 GB + GPU daemon load).

usage: python guard_mem.py <required_free_MB>   (in .cmd:  python guard_mem.py 12000 || exit /b 1)
"""
import ctypes, sys


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free_mb = m.ullAvailPhys // 2**20; need = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f"guard_mem: free {free_mb} MB, need {need} MB -> {'OK' if free_mb >= need else 'REFUSED'}", flush=True)
sys.exit(0 if free_mb >= need else 1)
