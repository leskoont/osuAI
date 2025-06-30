import mss
import numpy as np
import cv2
from PIL import Image
import time

class ScreenCapture:
    def __init__(self, monitor=None):
        """
        Инициализация захвата экрана
        :param monitor: Область захвата экрана (None = основной монитор)
        """
        self.sct = mss.mss()
        if monitor is None:
            self.monitor = self.sct.monitors[0]  # Основной монитор
        else:
            self.monitor = monitor
            
    def capture(self):
        """
        Захват экрана и преобразование в numpy массив
        :return: Изображение в формате numpy array (RGB)
        """
        sct_img = self.sct.grab(self.monitor)
        # Преобразование в RGB формат
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # Преобразование в numpy массив
        return np.array(img)
    
    def capture_preprocessed(self, target_size=(256, 256), grayscale=True):
        """
        Захват экрана с предобработкой для нейронной сети
        :param target_size: Целевой размер изображения (высота, ширина)
        :param grayscale: Преобразовать в оттенки серого
        :return: Предобработанное изображение
        """
        img = self.capture()
        
        # Изменение размера
        img = cv2.resize(img, target_size)
        
        # Преобразование в оттенки серого, если требуется
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)  # Добавление канала
            
        # Нормализация
        img = img / 255.0
        
        return img 