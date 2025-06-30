import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

class OsuNeuralNetwork:
    def __init__(self, input_shape=(256, 256, 1), learning_rate=0.001):
        """
        Инициализация нейронной сети для игры в Osu
        :param input_shape: Размерность входного изображения (высота, ширина, каналы)
        :param learning_rate: Скорость обучения
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.memory_buffer = []  # Буфер для хранения опыта
        self.max_buffer_size = 10000  # Максимальный размер буфера
        
    def _build_model(self):
        """
        Построение архитектуры нейронной сети
        :return: Скомпилированная модель
        """
        model = Sequential([
            # Сверточные слои для обработки изображения
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Полносвязные слои
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Выходной слой: [x, y, z_key, x_key]
            # x, y - координаты мыши (0-1)
            # z_key, x_key - вероятности нажатия клавиш (0-1)
            Dense(4, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        model.summary()
        return model
    
    def predict(self, state):
        """
        Предсказание действия по состоянию экрана
        :param state: Изображение экрана (после предобработки)
        :return: Предсказанное действие [x, y, z_key, x_key]
        """
        # Добавляем размерность батча, если нужно
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        
        return self.model.predict(state)[0]
    
    def add_to_memory(self, state, action, reward):
        """
        Добавление опыта в буфер памяти
        :param state: Состояние (изображение экрана)
        :param action: Действие [x, y, z_key, x_key]
        :param reward: Награда
        """
        # Если буфер полный, удаляем старые записи
        if len(self.memory_buffer) >= self.max_buffer_size:
            self.memory_buffer.pop(0)
            
        self.memory_buffer.append((state, action, reward))
    
    def train(self, batch_size=32, epochs=10):
        """
        Обучение модели на основе накопленного опыта
        :param batch_size: Размер батча
        :param epochs: Количество эпох
        :return: История обучения
        """
        if len(self.memory_buffer) < batch_size:
            print(f"Недостаточно данных для обучения. Текущий размер буфера: {len(self.memory_buffer)}")
            return None
        
        # Подготовка батча для обучения
        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        batch = [self.memory_buffer[i] for i in indices]
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        
        # Используем награды для коррекции действий
        target_actions = actions * rewards.reshape(-1, 1)
        
        # Обучение модели
        history = self.model.fit(states, target_actions, epochs=epochs, verbose=1)
        return history
    
    def save_model(self, filepath="models/osu_model.h5"):
        """
        Сохранение модели в файл
        :param filepath: Путь для сохранения
        """
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath="models/osu_model.h5"):
        """
        Загрузка модели из файла
        :param filepath: Путь к файлу модели
        """
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Модель загружена из {filepath}")
        else:
            print(f"Файл модели {filepath} не найден. Используется новая модель.") 