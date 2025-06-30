import time
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import cv2
import win32gui
import win32con
import threading
from screen_capture import ScreenCapture
from input_controller import InputController
from neural_network import OsuNeuralNetwork
from reward_system import RewardSystem

def is_window_foreground(window_title):
    """
    Проверяет, находится ли окно с указанным заголовком на переднем плане
    :param window_title: Заголовок окна или его часть
    :return: True если окно с этим заголовком на переднем плане
    """
    try:
        # Получаем окно на переднем плане
        foreground_hwnd = win32gui.GetForegroundWindow()
        # Получаем его заголовок
        fg_title = win32gui.GetWindowText(foreground_hwnd).lower()
        # Проверяем, содержит ли заголовок искомую строку
        return window_title.lower() in fg_title
    except Exception as e:
        print(f"Ошибка при проверке окна: {e}")
        return False

def set_foreground_window(window_title):
    """
    Устанавливает окно с заданным заголовком на передний план
    :param window_title: Заголовок окна или его часть
    :return: True, если окно найдено и установлено на передний план
    """
    try:
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                windows.append(hwnd)
                return True
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            # Нашли окно, устанавливаем его на передний план
            try:
                # Сначала попробуем отправить сообщение окну
                win32gui.SendMessage(windows[0], win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
                # Затем попробуем установить на передний план
                win32gui.SetForegroundWindow(windows[0])
                # Проверка, действительно ли окно на переднем плане
                time.sleep(0.1)  # Даем время на переключение
                if is_window_foreground(window_title):
                    return True
                else:
                    # Запасной вариант - используем альтернативный метод
                    win32gui.ShowWindow(windows[0], win32con.SW_RESTORE)
                    win32gui.BringWindowToTop(windows[0])
                    win32gui.SetForegroundWindow(windows[0])
                    return True
            except Exception as e:
                print(f"Предупреждение: Не удалось установить окно на передний план: {e}")
                print("Это нормально - Windows может ограничивать смену фокуса окон")
                return False
        return False
    except Exception as e:
        print(f"Ошибка при работе с окном: {e}")
        return False

# Функция для непрерывного обучения в отдельном потоке
def continuous_training(neural_network, stop_event, pause_event, save_event, save_interval=500):
    """
    Непрерывное обучение нейросети в отдельном потоке
    :param neural_network: Нейронная сеть для обучения
    :param stop_event: Событие для остановки обучения
    :param pause_event: Событие для паузы/продолжения обучения
    :param save_event: Событие для сохранения модели
    :param save_interval: Интервал между сохранениями модели (в эпохах)
    """
    print("Поток обучения запущен...")
    epoch_counter = 0
    
    while not stop_event.is_set():
        # Проверка паузы
        if pause_event.is_set():
            # Обучение приостановлено
            time.sleep(0.5)  # Ждем полсекунды перед следующей проверкой
            continue
            
        # Проверка наличия данных для обучения
        if len(neural_network.memory_buffer) < 32:
            print("Недостаточно данных для обучения. Сбор данных...")
            time.sleep(2)  # Ждем накопления данных
            continue
        
        # Обучение одной эпохи
        batch_size = min(32, len(neural_network.memory_buffer))
        start_time = time.time()
        history = neural_network.train(batch_size=batch_size, epochs=1)
        training_time = time.time() - start_time
        
        # Увеличиваем счетчик эпох
        epoch_counter += 1
        
        # Сохранение модели через указанные интервалы
        if epoch_counter % save_interval == 0 or save_event.is_set():
            neural_network.save_model()
            print(f"\nМодель сохранена после {epoch_counter} эпох обучения")
            save_event.clear()  # Сбрасываем флаг сохранения
            
        # Вывод статистики каждые 10 эпох
        if epoch_counter % 10 == 0:
            if history:
                loss = history.history.get('loss', [0])[0]
                print(f"Эпоха {epoch_counter}, Loss: {loss:.4f}, Время: {training_time:.2f}с, Размер буфера: {len(neural_network.memory_buffer)}")
            else:
                print(f"Эпоха {epoch_counter}, Время: {training_time:.2f}с, Размер буфера: {len(neural_network.memory_buffer)}")
        
        # Небольшая задержка между эпохами, чтобы не загружать CPU
        time.sleep(0.1)
    
    print("Поток обучения остановлен.")

def main():
    """
    Основная функция для запуска нейросети для игры в Osu
    """
    # Установка окна Osu на передний план
    if set_foreground_window("osu!"):
        print("Окно Osu! найдено и установлено на передний план")
    else:
        print("Окно Osu! не найдено или не может быть установлено на передний план.")
        print("Убедитесь, что игра запущена и находится в фокусе перед началом работы.")
    
    # Запрос у пользователя настройки клавиш в Osu
    print("\nВажно: настройте клавиши, используемые в Osu!")
    primary_key = input("Введите первую клавишу для нажатия в Osu (по умолчанию 'w'): ").strip().lower() or 'w'
    secondary_key = input("Введите вторую клавишу для нажатия в Osu (по умолчанию 'e'): ").strip().lower() or 'e'
    
    # Инициализация компонентов
    screen_width, screen_height = 2560, 1440  # Разрешение экрана
    
    # Выбор конкретного монитора для захвата
    import mss
    sct = mss.mss()
    print("Обнаружены мониторы:")
    for i, monitor in enumerate(sct.monitors):
        print(f"Монитор {i}: {monitor}")
    
    # Запрос у пользователя, какой монитор использовать
    monitor_index = 1  # По умолчанию первый физический монитор
    try:
        monitor_choice = input("Введите номер монитора для захвата (0 - все мониторы, 1 - первый, 2 - второй и т.д.): ")
        monitor_index = int(monitor_choice)
        if monitor_index < 0 or monitor_index >= len(sct.monitors):
            print(f"Неверный номер монитора. Используем первый физический монитор (индекс 1)")
            monitor_index = 1
    except ValueError:
        print("Введено некорректное значение. Используем первый физический монитор (индекс 1)")
    
    # Выбор монитора
    selected_monitor = sct.monitors[monitor_index]
    print(f"Выбран монитор {monitor_index}: {selected_monitor}")
    
    # Создаем объект захвата с выбранным монитором
    screen_capture = ScreenCapture(monitor=selected_monitor)
    input_controller = InputController(screen_width, screen_height)
    
    # Настраиваем клавиши в контроллере ввода
    input_controller.key_mapping = {
        primary_key: ord(primary_key.upper()),
        secondary_key: ord(secondary_key.upper())
    }
    # Устанавливаем имена клавиш для методов
    input_controller.primary_key = primary_key
    input_controller.secondary_key = secondary_key
    print(f"Настроены клавиши: {primary_key} и {secondary_key}")
    
    reward_system = RewardSystem()
    
    # Создание и загрузка нейронной сети
    neural_network = OsuNeuralNetwork()
    try:
        neural_network.load_model()
    except:
        print("Не удалось загрузить сохраненную модель. Используется новая модель.")
    
    # Параметры обучения
    save_interval = 500  # Интервал сохранения модели (в эпохах)
    train_on_start = True  # Обучать ли модель с самого начала
    
    # Создание окна для визуализации
    cv2.namedWindow("OsuAI Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OsuAI Vision", 320, 240)  # Маленькое окно
    
    # Статистика производительности
    rewards = []
    fps_values = []
    
    # Флаги управления
    running = True
    paused = True  # Программа начинается на паузе
    training = train_on_start
    
    # События для управления потоком обучения
    stop_training_event = threading.Event()  # Сигнал для остановки обучения
    pause_training_event = threading.Event()  # Сигнал для паузы/продолжения обучения
    save_model_event = threading.Event()  # Сигнал для сохранения модели
    
    # Устанавливаем начальное состояние обучения
    if not training:
        pause_training_event.set()  # Если training = False, ставим на паузу
    
    # Запускаем поток непрерывного обучения
    training_thread = threading.Thread(
        target=continuous_training, 
        args=(neural_network, stop_training_event, pause_training_event, save_model_event, save_interval)
    )
    training_thread.daemon = True  # Поток завершится вместе с основным потоком
    training_thread.start()
    
    print("Запуск ИИ для игры в Osu")
    print("ПРОГРАММА НА ПАУЗЕ. Нажмите 'p' для начала работы.")
    print("Нажмите 'p' для паузы/продолжения")
    print("Нажмите 't' для включения/выключения обучения")
    print("Нажмите 's' для принудительного сохранения модели")
    print("Нажмите 'f' для установки окна Osu на передний план")
    print("Нажмите 'q' для выхода")
    print("\nВАЖНО: перед запуском вручную переключитесь на окно Osu!")
    
    # Счетчик итераций
    iteration = 0
    
    # Предыдущий кадр для сравнения
    prev_frame = None
    
    # Убеждаемся, что Osu в фокусе перед началом (с обработкой ошибок)
    print("Подождите 3 секунды и вручную переключитесь на окно Osu...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    try:
        if not is_window_foreground("osu!"):
            print("Окно Osu не в фокусе. Пытаемся установить его на передний план...")
            set_foreground_window("osu!")
            # Проверяем еще раз
            if is_window_foreground("osu!"):
                print("Окно Osu! успешно установлено на передний план.")
            else:
                print("Не удалось установить окно Osu! на передний план. Пожалуйста, сделайте это вручную перед снятием паузы.")
        else:
            print("Окно Osu! уже в фокусе.")
    except Exception as e:
        print(f"Ошибка при проверке окна: {e}")
        print("Пожалуйста, убедитесь, что окно Osu! в фокусе перед снятием паузы.")
    
    # Главный цикл
    while running:
        start_time = time.time()
        
        # Обработка клавиш управления
        if keyboard.is_pressed('p'):
            paused = not paused
            print("Пауза" if paused else "Продолжение")
            
            # Если снимаем с паузы - убеждаемся, что Osu в фокусе
            if not paused:
                if not is_window_foreground("osu!"):
                    print("ВНИМАНИЕ: Окно Osu! не в фокусе! Действия могут не сработать. Переключитесь на окно игры.")
                    try:
                        set_foreground_window("osu!")
                    except:
                        pass
            
            time.sleep(0.3)  # Предотвращение многократного срабатывания
            
        if keyboard.is_pressed('t'):
            training = not training
            if training:
                pause_training_event.clear()  # Возобновляем обучение
                print("Обучение включено")
            else:
                pause_training_event.set()    # Приостанавливаем обучение
                print("Обучение выключено")
            time.sleep(0.3)  # Предотвращение многократного срабатывания
            
        if keyboard.is_pressed('s'):
            save_model_event.set()  # Устанавливаем событие сохранения модели
            print("Сохранение модели...")
            time.sleep(0.3)  # Предотвращение многократного срабатывания
            
        if keyboard.is_pressed('f'):
            try:
                if set_foreground_window("osu!"):
                    print("Окно Osu! установлено на передний план")
                else:
                    print("Не удалось установить окно Osu! на передний план. Сделайте это вручную.")
            except Exception as e:
                print(f"Ошибка при установке фокуса: {e}")
            time.sleep(0.3)  # Предотвращение многократного срабатывания
            
        if keyboard.is_pressed('q'):
            running = False
            continue
        
        if paused:
            # Даже на паузе продолжаем захватывать и показывать экран
            frame = screen_capture.capture()
            display_frame = frame.copy()
            
            # Отображение надписи о том, что программа на паузе
            cv2.putText(display_frame, "ПАУЗА", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            
            # Отображение статуса обучения
            train_status = "ОБУЧЕНИЕ: ВКЛ" if training else "ОБУЧЕНИЕ: ВЫКЛ"
            train_color = (0, 255, 0) if training else (0, 0, 255)  # Зеленый или красный
            cv2.putText(display_frame, train_status, (10, frame.shape[0] - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, train_color, 2)
            
            # Проверка активного окна
            osu_active = is_window_foreground("osu!")
            status = "Окно Osu АКТИВНО" if osu_active else "Окно Osu НЕ АКТИВНО"
            color = (0, 255, 0) if osu_active else (0, 0, 255)  # Зеленый или красный
            cv2.putText(display_frame, status, (10, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Отображение в окне
            cv2.imshow("OsuAI Vision", display_frame)
            cv2.waitKey(1)  # Обновление окна
            
            time.sleep(0.1)
            continue
        
        # Проверка активного окна
        osu_active = is_window_foreground("osu!")
        if not osu_active and iteration % 10 == 0:  # Не спамим каждый кадр
            print("ВНИМАНИЕ: Окно Osu! не активно! Действия могут не сработать.")
            # Попытка вернуть фокус каждые 100 итераций
            if iteration % 100 == 0:
                try:
                    set_foreground_window("osu!")
                except:
                    pass
        
        # Захват и предобработка экрана
        frame = screen_capture.capture()
        processed_frame = screen_capture.capture_preprocessed()
        
        # Визуализация того, что видит модель
        display_frame = frame.copy()
        
        # Предсказание действия
        action = neural_network.predict(processed_frame)
        
        # Отображение предсказанных координат курсора
        cursor_x = int(action[0] * frame.shape[1])
        cursor_y = int(action[1] * frame.shape[0])
        cv2.circle(display_frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)  # Зеленый круг - предсказанное положение
        
        # Отображение информации о нажатиях клавиш
        primary_pressed = f"{primary_key.upper()}: ON" if action[2] > 0.5 else f"{primary_key.upper()}: OFF"
        secondary_pressed = f"{secondary_key.upper()}: ON" if action[3] > 0.5 else f"{secondary_key.upper()}: OFF"
        cv2.putText(display_frame, primary_pressed, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, secondary_pressed, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображение статуса обучения
        train_status = "ОБУЧЕНИЕ: ВКЛ" if training else "ОБУЧЕНИЕ: ВЫКЛ"
        train_color = (0, 255, 0) if training else (0, 0, 255)  # Зеленый или красный
        cv2.putText(display_frame, train_status, (10, frame.shape[0] - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, train_color, 2)
        
        # Отображение статуса окна Osu
        status = "Osu АКТИВНО" if osu_active else "Osu НЕ АКТИВНО"
        color = (0, 255, 0) if osu_active else (0, 0, 255)  # Зеленый или красный
        cv2.putText(display_frame, status, (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Отображение обнаруженных кругов Osu
        circles = reward_system.detect_hit_circles(frame)
        approach_circles = reward_system.detect_approach_circles(frame)
        
        # Отладочная отрисовка обнаруженных объектов
        cursor_pos = (cursor_x, cursor_y)
        display_frame = reward_system.debug_draw_detection(display_frame, circles, approach_circles, cursor_pos)
        
        # Отображение дополнительной информации о кругах
        if circles:
            for i, circle in enumerate(circles[:3]):  # Показываем только первые 3 круга
                info_text = f"Circle {i+1}: {circle['color']} R:{circle['radius']} T:{circle['time_factor']:.2f}"
                cv2.putText(display_frame, info_text, (10, 110 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Отображение статистики игры из интерфейса
        game_stats = reward_system.parse_game_stats(frame)
        stats_text = f"Score: {game_stats['score']} | Combo: {game_stats['combo']} | Acc: {game_stats['accuracy']:.1f}%"
        cv2.putText(display_frame, stats_text, (10, frame.shape[0] - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Отображение в окне
        cv2.imshow("OsuAI Vision", display_frame)
        cv2.waitKey(1)  # Обновление окна
        
        # Выполнение действия только если окно Osu активно
        if osu_active:
            input_controller.execute_action(action)
        
        # Расчет вознаграждения
        reward = reward_system.calculate_reward(frame, action, prev_frame)
        rewards.append(reward)
        
        # Сохранение опыта для обучения
        neural_network.add_to_memory(processed_frame, action, reward)
        
        # Обновление счетчика итераций
        iteration += 1
        
        # Отображение статистики каждые 500 итераций
        if iteration % 500 == 0:
            # Вывод статистики
            stats = reward_system.get_stats()
            print(f"\nРасширенная статистика после {iteration} итераций:")
            print(f"Попаданий: {stats['hits']}, Промахов: {stats['misses']}")
            print(f"Текущее комбо: {stats['combo']}, Максимальное комбо: {stats['max_combo']}")
            print(f"Точность ИИ: {stats['accuracy']:.2f}%")
            print(f"Игровая точность: {stats['current_game_accuracy']:.2f}%")
            print(f"Текущий счет: {stats['current_score']}")
            print(f"Стабильность тайминга: {stats['timing_stability']:.1f}%")
            print(f"Средний счет: {stats['avg_recent_score']:.0f}")
            print(f"Всего действий: {stats['total_actions']}")
            
            # Вывод графика наград
            plt.figure(figsize=(10, 5))
            plt.plot(rewards[-100:])
            plt.title('Награды за последние 100 итераций')
            plt.xlabel('Итерация')
            plt.ylabel('Награда')
            plt.savefig('rewards.png')
            plt.close()
        
        # Сохранение предыдущего кадра
        prev_frame = frame
        
        # Расчет FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_values.append(fps)
        
        # Вывод текущего состояния (каждые 10 итераций)
        if iteration % 10 == 0:
            avg_fps = np.mean(fps_values[-10:])
            avg_reward = np.mean(rewards[-10:])
            print(f"Итерация: {iteration}, FPS: {avg_fps:.1f}, Награда: {avg_reward:.3f}", end='\r')
    
    # Завершение программы - останавливаем поток обучения
    print("\nЗавершение работы программы...")
    stop_training_event.set()
    training_thread.join(timeout=2)  # Ждем завершения потока не более 2 секунд
    
    # Закрытие окна визуализации
    cv2.destroyAllWindows()
    
    # Сохранение модели перед выходом
    print("Сохранение модели...")
    neural_network.save_model()
    
    # Вывод итоговой статистики
    print("\n\nИтоговая расширенная статистика:")
    stats = reward_system.get_stats()
    print(f"Попаданий: {stats['hits']}, Промахов: {stats['misses']}")
    print(f"Максимальное комбо: {stats['max_combo']}")
    print(f"Точность ИИ: {stats['accuracy']:.2f}%")
    print(f"Игровая точность: {stats['current_game_accuracy']:.2f}%")
    print(f"Финальный счет: {stats['current_score']}")
    print(f"Стабильность тайминга: {stats['timing_stability']:.1f}%")
    print(f"Всего действий: {stats['total_actions']}")
    print(f"Средний счет за сессию: {stats['avg_recent_score']:.0f}")
    
    # Вывод графика наград
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('История наград')
    plt.xlabel('Итерация')
    plt.ylabel('Награда')
    plt.savefig('rewards_total.png')
    
    # Вывод графика FPS
    plt.figure(figsize=(10, 5))
    plt.plot(fps_values)
    plt.title('История FPS')
    plt.xlabel('Итерация')
    plt.ylabel('FPS')
    plt.savefig('fps_total.png')
    
    print("Программа завершена")

if __name__ == "__main__":
    main() 