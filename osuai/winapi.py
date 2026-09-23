"""Тонкий слой WinAPI на ctypes (без pywin32/pydirectinput/keyboard).

Все функции безопасно деградируют на не-Windows системах (для тестов и симулятора).
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    winmm = ctypes.WinDLL("winmm")
    # Явные сигнатуры: по умолчанию ctypes считает всё 32-битным int, а HANDLE/HWND на x64 — 64 бит.
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                    ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79

ULONG_PTR = ctypes.c_size_t  # в исходной версии был POINTER(ULONG) — неверный тип для dwExtraInfo


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_VK_NAMED = {
    "space": 0x20, "enter": 0x0D, "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "tab": 0x09,
    "esc": 0x1B, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "mouse1": 0x01, "mouse2": 0x02,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}


def vk_code(name: str) -> int:
    name = name.strip().lower()
    if name in _VK_NAMED:
        return _VK_NAMED[name]
    if len(name) == 1 and (name.isalpha() or name.isdigit()):
        return ord(name.upper())
    raise ValueError(f"Неизвестная клавиша: {name!r}")


# ---------------------------------------------------------------------------- system
def set_dpi_aware() -> None:
    """Без этого на мониторах с масштабированием 125-150% координаты окна/курсора
    приходят в «логических» пикселях и не совпадают со снимком экрана."""
    if not IS_WINDOWS:
        return
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            user32.SetProcessDPIAware()


def begin_timer_resolution(ms: int) -> None:
    if IS_WINDOWS and ms > 0:
        winmm.timeBeginPeriod(ms)


def end_timer_resolution(ms: int) -> None:
    if IS_WINDOWS and ms > 0:
        winmm.timeEndPeriod(ms)


def set_process_priority(level: str) -> None:
    if not IS_WINDOWS:
        return
    classes = {"normal": 0x20, "above_normal": 0x8000, "high": 0x80}
    if level in classes:
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), classes[level])


# ---------------------------------------------------------------------------- windows
def _process_name(pid: int) -> str:
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(h)


def find_window(title: str, process_name: str = "") -> int | None:
    """Ищет видимое окно игры. Проверка имени процесса отсекает, например, вкладку
    браузера с «osu!» в заголовке (исходная версия могла кликать по браузеру)."""
    if not IS_WINDOWS:
        return None
    found: list[tuple[int, int]] = []
    title_l = title.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        text = buf.value.lower()
        if title_l in text:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if process_name and _process_name(pid.value).lower() != process_name.lower():
                return True
            found.append((0 if text.startswith(title_l) else 1, hwnd))
        return True

    user32.EnumWindows(cb, 0)
    return min(found)[1] if found else None


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom


def monitor_rect_for(hwnd: int) -> tuple[int, int, int, int]:
    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
    hmon = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(hmon, ctypes.byref(info))
    r = info.rcMonitor
    return r.left, r.top, r.right, r.bottom


def foreground_window() -> int | None:
    return user32.GetForegroundWindow() if IS_WINDOWS else None


def focus_window(hwnd: int) -> bool:
    if not IS_WINDOWS or not hwnd:
        return False
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    # Windows запрещает SetForegroundWindow фоновым процессам; «нажатие» Alt снимает запрет.
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0)
    user32.BringWindowToTop(hwnd)
    return bool(user32.SetForegroundWindow(hwnd))


def virtual_screen() -> tuple[int, int, int, int]:
    if not IS_WINDOWS:
        return 0, 0, 1920, 1080
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN), user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    if IS_WINDOWS:
        user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def key_down(vk: int) -> bool:
    return bool(IS_WINDOWS and user32.GetAsyncKeyState(vk) & 0x8000)


def scan_code(vk: int) -> int:
    return user32.MapVirtualKeyW(vk, 0) if IS_WINDOWS else 0


def send_inputs(inputs: list[INPUT]) -> None:
    if not IS_WINDOWS or not inputs:
        return
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
