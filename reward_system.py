import cv2
import numpy as np
import time
import pytesseract
from collections import deque

class RewardSystem:
    def __init__(self):
        """
        Инициализация улучшенной системы вознаграждений для Osu
        """
        # Цвета объектов в Osu (в формате BGR для OpenCV)
        self.hit_circle_colors = {
            'pink': ([145, 100, 100], [165, 255, 255]),  # Розовый в HSV
            'blue': ([100, 100, 100], [120, 255, 255]),  # Синий в HSV
            'green': ([40, 100, 100], [80, 255, 255]),   # Зеленый в HSV
            'yellow': ([20, 100, 100], [30, 255, 255]),  # Желтый в HSV
        }
        
        # Области интерфейса (координаты в процентах от размера экрана)
        self.ui_regions = {
            'score': (0.0, 0.0, 0.3, 0.15),      # Левый верхний угол - счет
            'combo': (0.0, 0.85, 0.3, 1.0),      # Левый нижний угол - комбо
            'accuracy': (0.7, 0.0, 1.0, 0.15),   # Правый верхний угол - точность
            'hp_bar': (0.0, 0.15, 0.02, 0.85),   # Левая сторона - полоса HP
        }
        
        # Счетчики и статистика
        self.hit_count = 0
        self.miss_count = 0
        self.combo = 0
        self.max_combo = 0
        self.current_score = 0
        self.current_accuracy = 100.0
        
        # История для анализа изменений
        self.score_history = deque(maxlen=10)
        self.combo_history = deque(maxlen=10)
        self.accuracy_history = deque(maxlen=10)
        
        # Временные метки для анализа ритма
        self.last_hit_time = 0
        self.hit_timing_buffer = deque(maxlen=50)
        
        # Параметры для анализа кругов
        self.circle_detection_params = {
            'min_radius': 15,
            'max_radius': 100,
            'min_area': 500,
            'approach_circle_threshold': 0.8
        }
        
        # Веса для различных компонентов награды
        self.reward_weights = {
            'hit_accuracy': 0.4,      # Точность попадания
            'timing': 0.3,            # Тайминг нажатия
            'combo_bonus': 0.2,       # Бонус за комбо
            'score_increase': 0.1     # Увеличение счета
        }

    def extract_ui_text(self, frame, region_name):
        """
        Извлечение текста из определенной области интерфейса
        :param frame: Кадр игры
        :param region_name: Название области ('score', 'combo', 'accuracy')
        :return: Извлеченный текст
        """
        if region_name not in self.ui_regions:
            return ""
        
        # Получаем координаты области
        x1_pct, y1_pct, x2_pct, y2_pct = self.ui_regions[region_name]
        h, w = frame.shape[:2]
        
        x1, y1 = int(x1_pct * w), int(y1_pct * h)
        x2, y2 = int(x2_pct * w), int(y2_pct * h)
        
        # Вырезаем область
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Предобработка для лучшего распознавания текста
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Увеличиваем контраст
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Увеличиваем изображение для лучшего распознавания
        binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        try:
            # Настройки для распознавания цифр
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,%x'
            text = pytesseract.image_to_string(binary, config=config).strip()
            return text
        except:
            return ""

    def parse_game_stats(self, frame):
        """
        Парсинг игровой статистики из интерфейса
        :param frame: Кадр игры
        :return: Словарь с игровой статистикой
        """
        stats = {
            'score': self.current_score,
            'combo': self.combo,
            'accuracy': self.current_accuracy
        }
        
        # Извлечение счета
        score_text = self.extract_ui_text(frame, 'score')
        try:
            # Убираем все символы кроме цифр
            score_digits = ''.join(filter(str.isdigit, score_text))
            if score_digits:
                stats['score'] = int(score_digits)
        except:
            pass
        
        # Извлечение комбо
        combo_text = self.extract_ui_text(frame, 'combo')
        try:
            combo_digits = ''.join(filter(str.isdigit, combo_text))
            if combo_digits:
                stats['combo'] = int(combo_digits)
        except:
            pass
        
        # Извлечение точности
        accuracy_text = self.extract_ui_text(frame, 'accuracy')
        try:
            # Ищем паттерн типа "99.50%"
            import re
            accuracy_match = re.search(r'(\d+\.?\d*)%?', accuracy_text)
            if accuracy_match:
                stats['accuracy'] = float(accuracy_match.group(1))
        except:
            pass
        
        return stats

    def detect_hit_circles(self, frame):
        """
        Улучшенное обнаружение кругов для нажатия
        :param frame: Кадр игры (BGR)
        :return: Список кругов с дополнительной информацией
        """
        circles = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Обнаружение кругов разных цветов
        for color_name, (lower, upper) in self.hit_circle_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Создание маски для цвета
            mask = cv2.inRange(hsv, lower, upper)
            
            # Морфологические операции для очистки маски
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Поиск контуров
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.circle_detection_params['min_area']:
                    continue
                
                # Аппроксимация окружностью
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                if (self.circle_detection_params['min_radius'] <= radius <= 
                    self.circle_detection_params['max_radius']):
                    
                    # Оценка времени до нажатия на основе размера круга
                    # Большие круги = больше времени до нажатия
                    time_factor = radius / self.circle_detection_params['max_radius']
                    
                    circle_info = {
                        'center': center,
                        'radius': radius,
                        'color': color_name,
                        'time_factor': time_factor,
                        'area': area
                    }
                    circles.append(circle_info)
        
        return circles

    def detect_approach_circles(self, frame):
        """
        Обнаружение approach circles (белые кольца, показывающие тайминг)
        :param frame: Кадр игры
        :return: Список approach circles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение кругов методом Хафа
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=120
        )
        
        approach_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Проверяем, что это именно approach circle (белый/светлый)
                roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                          max(0, x-r):min(gray.shape[1], x+r)]
                if roi.size > 0:
                    mean_intensity = np.mean(roi)
                    if mean_intensity > 200:  # Белый/светлый круг
                        approach_circles.append({
                            'center': (x, y),
                            'radius': r,
                            'intensity': mean_intensity
                        })
        
        return approach_circles

    def calculate_hit_timing(self, circles, approach_circles, cursor_pos):
        """
        Расчет качества тайминга нажатия
        :param circles: Обнаруженные hit circles
        :param approach_circles: Обнаруженные approach circles
        :param cursor_pos: Позиция курсора (x, y)
        :return: Оценка тайминга (0-1)
        """
        if not circles or not approach_circles:
            return 0.5
        
        best_timing = 0
        
        for circle in circles:
            circle_center = circle['center']
            circle_radius = circle['radius']
            
            # Находим соответствующий approach circle
            for approach in approach_circles:
                approach_center = approach['center']
                approach_radius = approach['radius']
                
                # Проверяем, что центры близко
                center_distance = np.sqrt(
                    (circle_center[0] - approach_center[0])**2 + 
                    (circle_center[1] - approach_center[1])**2
                )
                
                if center_distance < 20:  # Центры близко
                    # Оценка тайминга на основе размера approach circle
                    # Чем меньше approach circle, тем ближе время нажатия
                    timing_factor = 1.0 - (approach_radius / 120.0)
                    timing_factor = max(0, min(1, timing_factor))
                    
                    # Проверяем, находится ли курсор в области круга
                    cursor_distance = np.sqrt(
                        (cursor_pos[0] - circle_center[0])**2 + 
                        (cursor_pos[1] - circle_center[1])**2
                    )
                    
                    if cursor_distance <= circle_radius:
                        best_timing = max(best_timing, timing_factor)
        
        return best_timing

    def calculate_reward(self, frame, action, prev_frame=None):
        """
        Улучшенный расчет вознаграждения
        :param frame: Текущий кадр игры
        :param action: Выполненное действие [x, y, key1, key2]
        :param prev_frame: Предыдущий кадр
        :return: Вознаграждение (0-1)
        """
        # Парсинг игровой статистики
        current_stats = self.parse_game_stats(frame)
        
        # Обнаружение игровых объектов
        circles = self.detect_hit_circles(frame)
        approach_circles = self.detect_approach_circles(frame)
        
        # Позиция курсора
        cursor_x = action[0] * frame.shape[1]
        cursor_y = action[1] * frame.shape[0]
        cursor_pos = (cursor_x, cursor_y)
        
        # Проверка нажатия клавиш
        key_pressed = action[2] > 0.5 or action[3] > 0.5
        
        # Компоненты награды
        reward_components = {
            'hit_accuracy': 0,
            'timing': 0,
            'combo_bonus': 0,
            'score_increase': 0
        }
        
        # 1. Анализ точности попадания
        if circles:
            best_hit_reward = 0
            for circle in circles:
                center = circle['center']
                radius = circle['radius']
                
                distance = np.sqrt((center[0] - cursor_x)**2 + (center[1] - cursor_y)**2)
                
                if distance <= radius:
                    # Точность попадания (чем ближе к центру, тем лучше)
                    accuracy = 1.0 - (distance / radius)
                    
                    if key_pressed:
                        # Отличное попадание с нажатием клавиши
                        hit_reward = 0.8 + (accuracy * 0.2)
                        best_hit_reward = max(best_hit_reward, hit_reward)
                        
                        # Обновляем статистику
                        self.hit_count += 1
                        current_time = time.time()
                        if self.last_hit_time > 0:
                            hit_interval = current_time - self.last_hit_time
                            self.hit_timing_buffer.append(hit_interval)
                        self.last_hit_time = current_time
                        
                    else:
                        # Хорошее позиционирование, но нет нажатия
                        best_hit_reward = max(best_hit_reward, 0.3 + (accuracy * 0.2))
            
            reward_components['hit_accuracy'] = best_hit_reward
            
            # Штраф за неправильное нажатие
            if key_pressed and best_hit_reward < 0.5:
                reward_components['hit_accuracy'] = 0
                self.miss_count += 1
        else:
            # Нет кругов - штраф за ненужное нажатие
            if key_pressed:
                reward_components['hit_accuracy'] = 0
                self.miss_count += 1
            else:
                reward_components['hit_accuracy'] = 0.1  # Небольшая награда за ожидание
        
        # 2. Анализ тайминга
        timing_quality = self.calculate_hit_timing(circles, approach_circles, cursor_pos)
        reward_components['timing'] = timing_quality
        
        # 3. Бонус за комбо
        combo_change = current_stats['combo'] - self.combo
        if combo_change > 0:
            # Комбо увеличилось
            combo_multiplier = min(current_stats['combo'] / 100.0, 2.0)  # Максимум x2
            reward_components['combo_bonus'] = 0.5 * combo_multiplier
        elif combo_change < 0:
            # Комбо сброшено
            reward_components['combo_bonus'] = -0.3
        
        # 4. Увеличение счета
        score_change = current_stats['score'] - self.current_score
        if score_change > 0:
            # Нормализуем увеличение счета
            score_reward = min(score_change / 10000.0, 0.5)
            reward_components['score_increase'] = score_reward
        
        # Обновляем внутреннюю статистику
        self.combo = current_stats['combo']
        self.current_score = current_stats['score']
        self.current_accuracy = current_stats['accuracy']
        
        if self.combo > self.max_combo:
            self.max_combo = self.combo
        
        # Добавляем в историю
        self.score_history.append(self.current_score)
        self.combo_history.append(self.combo)
        self.accuracy_history.append(self.current_accuracy)
        
        # Итоговая награда как взвешенная сумма компонентов
        total_reward = sum(
            reward_components[component] * self.reward_weights[component]
            for component in reward_components
        )
        
        # Ограничиваем награду диапазоном [0, 1]
        total_reward = max(0, min(1, total_reward))
        
        return total_reward

    def get_stats(self):
        """
        Получение расширенной статистики игры
        :return: Словарь со статистикой
        """
        total_notes = max(1, self.hit_count + self.miss_count)
        accuracy = self.hit_count / total_notes * 100
        
        # Анализ стабильности тайминга
        timing_stability = 0
        if len(self.hit_timing_buffer) > 5:
            timing_std = np.std(list(self.hit_timing_buffer))
            timing_stability = max(0, 100 - (timing_std * 100))
        
        # Средний счет за последние действия
        avg_recent_score = np.mean(list(self.score_history)) if self.score_history else 0
        
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'combo': self.combo,
            'max_combo': self.max_combo,
            'accuracy': accuracy,
            'current_score': self.current_score,
            'current_game_accuracy': self.current_accuracy,
            'timing_stability': timing_stability,
            'avg_recent_score': avg_recent_score,
            'total_actions': total_notes
        }

    def debug_draw_detection(self, frame, circles, approach_circles, cursor_pos):
        """
        Отрисовка обнаруженных объектов для отладки
        :param frame: Кадр для отрисовки
        :param circles: Обнаруженные hit circles
        :param approach_circles: Обнаруженные approach circles
        :param cursor_pos: Позиция курсора
        :return: Кадр с отрисованными объектами
        """
        debug_frame = frame.copy()
        
        # Отрисовка hit circles
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            color = (255, 0, 255) if circle['color'] == 'pink' else (0, 255, 0)
            cv2.circle(debug_frame, center, radius, color, 2)
            cv2.putText(debug_frame, circle['color'], 
                       (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Отрисовка approach circles
        for approach in approach_circles:
            center = approach['center']
            radius = approach['radius']
            cv2.circle(debug_frame, center, radius, (255, 255, 255), 1)
        
        # Отрисовка курсора
        cv2.circle(debug_frame, (int(cursor_pos[0]), int(cursor_pos[1])), 5, (0, 0, 255), -1)
        
        return debug_frame 