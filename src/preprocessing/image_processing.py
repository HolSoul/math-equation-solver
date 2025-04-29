# Функции для предобработки изображений
# Например: загрузка, изменение размера, бинаризация, нормализация, аугментация

import cv2
import numpy as np
import os
import torch

def preprocess_image(image_path: str, target_height: int = 64) -> torch.Tensor | None:
    """Загружает, предобрабатывает изображение для модели распознавания.

    Шаги:
    1. Загрузка изображения в оттенках серого.
    2. Бинаризация с использованием инвертированного порога Оцу.
    3. Изменение размера до target_height, сохраняя пропорции.
    4. Нормализация значений пикселей в диапазон [0, 1].
    5. Преобразование в тензор PyTorch формата (C, H, W).

    Args:
        image_path: Путь к файлу изображения.
        target_height: Целевая высота изображения после изменения размера.

    Returns:
        Предобработанное изображение PyTorch Tensor (float32) или None, если обработка не удалась.
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл не найден по пути: {image_path}")
        return None

    try:
        # 1. Загрузка в оттенках серого
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"Предупреждение: Не удалось загрузить изображение {image_path}, возможно, некорректный файл.")
            return None

        # 2. Бинаризация (инвертированная, метод Оцу)
        _, image_binary = cv2.threshold(
            image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3. Изменение размера с сохранением пропорций
        h, w = image_binary.shape
        if h == 0 or w == 0:
            print(f"Предупреждение: Изображение {image_path} имеет нулевую высоту или ширину после загрузки.")
            return None
        aspect_ratio = w / h
        target_width = max(1, int(target_height * aspect_ratio)) # Убедимся, что ширина >= 1

        interpolation = cv2.INTER_AREA if target_height < h else cv2.INTER_LINEAR
        image_resized = cv2.resize(image_binary, (target_width, target_height), interpolation=interpolation)

        # 4. Нормализация в диапазон [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # 5. Преобразование в тензор PyTorch формата (C, H, W)
        # Добавляем измерение канала: (H, W) -> (H, W, 1)
        image_numpy = np.expand_dims(image_normalized, axis=-1)
        # Преобразуем в тензор PyTorch
        image_tensor = torch.from_numpy(image_numpy)
        # Меняем порядок измерений: (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

    except Exception as e:
        print(f"Ошибка при предобработке изображения {image_path}: {e}")
        return None

# Пример использования (можно раскомментировать для быстрой проверки)
# if __name__ == '__main__':
#     # Убедитесь, что этот путь существует или замените его
#     test_image_path = '../../notebooks/sample_images/sample_equation.png'
#     processed = preprocess_image(test_image_path, target_height=64)
#     if processed is not None:
#         print(f"Предобработка завершена. Форма результата: {processed.shape}, Тип: {processed.dtype}")
#         # Показать результат (нужно изменить порядок измерений для imshow)
#         import matplotlib.pyplot as plt
#         plt.imshow(processed.permute(1, 2, 0).squeeze(), cmap='gray') # (C,H,W) -> (H,W,C)
#         plt.title("Предобработанное изображение")
#         plt.show()

print("Файл для функций предобработки изображений") 