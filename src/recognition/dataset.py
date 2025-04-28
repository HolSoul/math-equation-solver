import torch
from torch.utils.data import Dataset
import os
from PIL import Image # Используем PIL для большей совместимости форматов
import cv2 # OpenCV может понадобиться для некоторых операций

# Импортируем наши модули
from src.preprocessing.image_processing import preprocess_image
from src.recognition.utils import Vocabulary # Предполагается, что Vocabulary уже создан и передан

class LatexDataset(Dataset):
    """Кастомный Dataset для загрузки изображений формул и их LaTeX представления."""

    def __init__(self, image_paths, formulas, vocab: Vocabulary, target_height=64, transform=None):
        """Инициализация датасета.

        Args:
            image_paths (list): Список путей к файлам изображений.
            formulas (list): Список соответствующих LaTeX формул (строки).
            vocab (Vocabulary): Экземпляр класса Vocabulary, уже построенный или загруженный.
            target_height (int): Целевая высота изображений после предобработки.
            transform (callable, optional): Дополнительные трансформации для изображения (аугментация и т.д.).
                                             Пока не используется нашей preprocess_image, но можно добавить.
        """
        if len(image_paths) != len(formulas):
            raise ValueError("Количество путей к изображениям и формул должно совпадать!")
        
        self.image_paths = image_paths
        self.formulas = formulas
        self.vocab = vocab
        self.target_height = target_height
        self.transform = transform # Для будущих аугментаций
        self.preprocess_fn = preprocess_image # Ссылка на нашу функцию

        print(f"Создан LatexDataset с {len(self)} образцами.")

    def __len__(self):
        """Возвращает общее количество образцов в датасете."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Загружает, предобрабатывает и возвращает один образец (изображение, формула).

        Args:
            idx (int): Индекс образца.

        Returns:
            tuple: Кортеж (image_tensor, numericalized_formula)
                   image_tensor: Предобработанное изображение (torch.Tensor).
                   numericalized_formula: Нумеризованная формула (list of int).
                   В collate_fn она будет преобразована в тензор.
                   Если изображение не удалось загрузить/обработать, возвращает None.
        """
        img_path = self.image_paths[idx]
        formula_str = self.formulas[idx]

        # 1. Обработка изображения
        try:
            # Используем нашу функцию предобработки
            # Она включает загрузку, перевод в серое, бинаризацию, ресайз, нормализацию
            image_tensor = self.preprocess_fn(img_path, target_height=self.target_height)
            if image_tensor is None:
                print(f"Предупреждение: Не удалось обработать изображение {img_path}. Пропускаем образец {idx}.")
                # Вернем None или обработаем иначе, чтобы DataLoader мог пропустить
                # Для простоты пока вернем фиктивные данные или вызовем исключение, 
                # но лучше реализовать пропуск в collate_fn или настроить DataLoader.
                # Пока вернем None, чтобы указать на проблему.
                return None 

            # Применение дополнительных трансформаций (если они есть)
            if self.transform:
                # PIL Image может быть нужен для torchvision.transforms
                # image = Image.open(img_path).convert('L') 
                # image_tensor = self.transform(image)
                pass # Пока не реализовано
        
        except Exception as e:
            print(f"Ошибка при обработке изображения {img_path} (индекс {idx}): {e}")
            return None

        # 2. Обработка формулы
        try:
            numericalized_formula = self.vocab.numericalize(formula_str)
        except Exception as e:
            print(f"Ошибка при нумеризации формулы '{formula_str}' (индекс {idx}): {e}")
            return None # Пропускаем образец, если формула не может быть обработана

        return image_tensor, numericalized_formula

# --- Пример использования (закомментирован) ---
# Нужен построенный словарь и реальные данные

# if __name__ == '__main__':
#     from utils import Vocabulary, collate_fn
#     from torch.utils.data import DataLoader
# 
#     # 1. Создать или загрузить словарь
#     # Предположим, у нас есть список всех формул из датасета
#     all_formulas = ["a=1", "x^{2}+y^{2}=z^{2}", "b", "\sin(x)=0"] # Заменить реальными!
#     vocab = Vocabulary(freq_threshold=1)
#     vocab.build_vocabulary(all_formulas)
#     VOCAB_SIZE = len(vocab)
#     print(f"Размер словаря: {VOCAB_SIZE}")
# 
#     # 2. Подготовить данные (пути и формулы)
#     # Замените на реальные пути!
#     img_dir = "../../data/images/" # Пример
#     image_paths = [os.path.join(img_dir, "img1.png"), os.path.join(img_dir, "img2.png"), os.path.join(img_dir, "img3.png"), os.path.join(img_dir, "img4.png")]
#     formulas = ["a=1", "x^{2}+y^{2}=z^{2}", "b", "\sin(x)=0"]
# 
#     # Проверка существования файлов (упрощенная)
#     # image_paths = [p for p in image_paths if os.path.exists(p)]
#     # formulas = [f for i, f in enumerate(formulas) if os.path.exists(image_paths[i])] # Убедиться, что соответствие сохранено!
     
#     if not image_paths:
#         print("Ошибка: Не найдены файлы изображений для теста Dataset.")
#     else:
#         # 3. Создать Dataset
#         dataset = LatexDataset(image_paths=image_paths, formulas=formulas, vocab=vocab, target_height=64)
# 
#         # 4. Создать DataLoader
#         # batch_size=2
#         # num_workers=0 # Для Windows часто лучше 0
#         # shuffle=True # Перемешивать для обучения
#         # collate_fn=collate_fn # Наша функция для обработки батчей
# 
#         # loader = DataLoader(
#         #     dataset=dataset,
#         #     batch_size=batch_size,
#         #     shuffle=shuffle,
#         #     num_workers=num_workers,
#         #     collate_fn=collate_fn
#         # )
# 
#         # 5. Пример итерации по DataLoader
#         # print("\nПример батча из DataLoader:")
#         # try:
#         #     # Получаем первый батч
#         #     for i, batch_data in enumerate(loader):
#         #         if batch_data is None: # Пропускаем, если __getitem__ вернул None
#         #             print(f"Пропущен None батч {i}")
#         #             continue
                 
#         #         images_b, formulas_b, lengths_b = batch_data
#         #         print(f"--- Батч {i+1} ---")
#         #         print(f"Форма тензора изображений: {images_b.shape}")
#         #         print(f"Форма тензора формул: {formulas_b.shape}")
#         #         print(f"Длины формул: {lengths_b.tolist()}")
#         #         # print(f"Формулы (индексы):\n{formulas_b}")
#         #         break # Показываем только первый батч
#         # except Exception as e:
#         #      print(f"Ошибка при итерации DataLoader: {e}")
#         #      import traceback
#         #      traceback.print_exc() 