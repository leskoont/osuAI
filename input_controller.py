import pydirectinput
import time
import ctypes
import win32api
import win32con
from ctypes import wintypes

# Константы для SendInput
INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Константы для мыши
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

# Структуры для SendInput
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]

class InputController:
    def __init__(self, screen_width=1920, screen_height=1080):
        """
        Инициализация контроллера ввода
        :param screen_width: Ширина экрана
        :param screen_height: Высота экрана
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Инициализация pydirectinput
        pydirectinput.PAUSE = 0  # Убираем задержку между командами
        
        # Словарь кодов клавиш и их имен
        self.key_mapping = {
            'w': 0x57,  # Виртуальный код клавиши W
            'e': 0x45   # Виртуальный код клавиши E
        }
        
        # Имена клавиш для выходов нейросети
        self.primary_key = 'w'    # Первая клавиша (индекс 2 в выходе)
        self.secondary_key = 'e'  # Вторая клавиша (индекс 3 в выходе)
        
        # Для совместимости с играми требуется использовать нормализованные координаты
        # для абсолютного позиционирования (0-65535)
        self.mouse_x = 0
        self.mouse_y = 0
        
    def move_mouse(self, x, y):
        """
        Перемещение мыши в указанные координаты
        :param x: X координата (0-1)
        :param y: Y координата (0-1)
        """
        # Преобразование нормализованных координат в реальные пиксели
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)
        
        # Сохраняем текущие координаты
        self.mouse_x = screen_x
        self.mouse_y = screen_y
        
        # Метод 1: Использование mouse_event
        # Преобразование в абсолютные координаты для системы (0-65535)
        abs_x = int(65535 * (screen_x / ctypes.windll.user32.GetSystemMetrics(0)))
        abs_y = int(65535 * (screen_y / ctypes.windll.user32.GetSystemMetrics(1)))
        
        # Перемещение мыши
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        
        # Метод 2: Использование SendInput (более надежный для некоторых игр)
        self.send_mouse_input(screen_x, screen_y)
        
    def send_mouse_input(self, x, y):
        """
        Отправляет событие перемещения мыши через SendInput API
        :param x: X координата в пикселях
        :param y: Y координата в пикселях
        """
        # Преобразование в абсолютные координаты для системы (0-65535)
        abs_x = int(65535 * (x / ctypes.windll.user32.GetSystemMetrics(0)))
        abs_y = int(65535 * (y / ctypes.windll.user32.GetSystemMetrics(1)))
        
        # Подготовка структуры ввода для SendInput
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(
            abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, ctypes.pointer(extra)
        )
        x = INPUT(INPUT_MOUSE, ii_)
        
        # Отправка события в систему
        ctypes.windll.user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
        
    def move_mouse_rel(self, dx, dy):
        """
        Относительное перемещение мыши
        :param dx: Изменение X (пиксели)
        :param dy: Изменение Y (пиксели)
        """
        # Метод 1: Использование mouse_event для относительного перемещения
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
        
        # Обновляем координаты
        self.mouse_x += dx
        self.mouse_y += dy
        
        # Метод 2: Использование SendInput для относительного перемещения
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        x = INPUT(INPUT_MOUSE, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
        
    def click(self, button='left'):
        """
        Нажатие кнопки мыши
        :param button: Кнопка для нажатия ('left' или 'right')
        """
        if button == 'left':
            # Метод 1: Использование mouse_event
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
            time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            
            # Метод 2: Использование SendInput
            extra = ctypes.c_ulong(0)
            ii_ = INPUT_UNION()
            ii_.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
            down = INPUT(INPUT_MOUSE, ii_)
            ii_.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
            up = INPUT(INPUT_MOUSE, ii_)
            
            ctypes.windll.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(down))
            time.sleep(0.01)
            ctypes.windll.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(up))
            
        elif button == 'right':
            # Аналогично для правой кнопки мыши
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
            time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)
    
    def press_key(self, key):
        """
        Нажатие клавиши клавиатуры (с использованием win32api)
        :param key: Клавиша для нажатия (например, 'w', 'e')
        """
        if key in self.key_mapping:
            # Получаем виртуальный код клавиши
            vk_code = self.key_mapping[key]
            
            # Симуляция нажатия и отпускания клавиши через SendInput
            self.send_key(vk_code, False)  # Нажатие
            time.sleep(0.01)
            self.send_key(vk_code, True)   # Отпускание
    
    def send_key(self, vk_code, up=False):
        """
        Отправляет событие нажатия клавиши используя SendInput API
        :param vk_code: Виртуальный код клавиши
        :param up: True для отпускания клавиши, False для нажатия
        """
        # Подготовка структуры ввода для SendInput
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.ki = KEYBDINPUT(vk_code, 0, KEYEVENTF_KEYUP if up else 0, 0, ctypes.pointer(extra))
        x = INPUT(INPUT_KEYBOARD, ii_)
        
        # Отправка события в систему
        ctypes.windll.user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
        
    def execute_action(self, action):
        """
        Выполнение действия на основе выхода модели
        :param action: Массив действий [x, y, primary_key, secondary_key]
                      x, y - координаты мыши (0-1)
                      primary_key, secondary_key - нажатия клавиш (0-1)
        """
        # Перемещение мыши (пробуем несколько методов, чтобы хотя бы один сработал)
        # Абсолютное перемещение
        self.move_mouse(action[0], action[1])
        
        # Добавляем небольшую задержку для обработки перемещения мыши
        time.sleep(0.02)
        
        # Также попробуем прямое использование pydirectinput (может работать лучше с некоторыми играми)
        screen_x = int(action[0] * self.screen_width)
        screen_y = int(action[1] * self.screen_height)
        pydirectinput.moveTo(screen_x, screen_y)
        
        # Нажатие клавиш, если вероятность больше 0.5
        if action[2] > 0.5:
            self.press_key(self.primary_key)
            
        if action[3] > 0.5:
            self.press_key(self.secondary_key) 