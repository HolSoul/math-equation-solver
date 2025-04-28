import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re

# --- Специальные токены ---
PAD_TOKEN = "<pad>"  # Токен для заполнения (padding)
SOS_TOKEN = "<sos>"  # Токен начала последовательности (start of sequence)
EOS_TOKEN = "<eos>"  # Токен конца последовательности (end of sequence)
UNK_TOKEN = "<unk>"  # Токен для неизвестных символов (на всякий случай)

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = SPECIAL_TOKENS.index(PAD_TOKEN)
SOS_IDX = SPECIAL_TOKENS.index(SOS_TOKEN)
EOS_IDX = SPECIAL_TOKENS.index(EOS_TOKEN)
UNK_IDX = SPECIAL_TOKENS.index(UNK_TOKEN)

class Vocabulary:
    def __init__(self, freq_threshold=1):
        """Инициализирует словарь.

        Args:
            freq_threshold: Минимальная частота символа для добавления в словарь.
        """
        # Начинаем индексацию со спец. токенов
        self.itos = {i: s for i, s in enumerate(SPECIAL_TOKENS)}
        self.stoi = {s: i for i, s in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.char_freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, formula_list):
        """Строит словарь на основе списка формул (LaTeX строк)."""
        print("Построение словаря...")
        # Сначала считаем частоты всех символов/токенов
        for formula in formula_list:
            # Простая токенизация: разбиваем на отдельные символы или LaTeX команды
            # ({}, \alpha, \frac, +, -, =, 0-9, a-z, A-Z, ...) - это нужно улучшить!
            tokens = self._basic_tokenizer(formula)
            self.char_freqs.update(tokens)

        # Добавляем токены, которые встречаются достаточно часто
        idx = len(self.itos) # Начинаем добавлять после спец. токенов
        for char, freq in self.char_freqs.items():
            if freq >= self.freq_threshold:
                if char not in self.stoi:
                    self.stoi[char] = idx
                    self.itos[idx] = char
                    idx += 1
        print(f"Словарь построен. Размер: {len(self)} токенов.")

    def _basic_tokenizer(self, formula):
        """Очень простая токенизация LaTeX. Требует значительного улучшения.
        Разбивает на отдельные символы или команды вида \cmd.
        """
        # Заменяем пробелы, чтобы они не потерялись
        formula = formula.replace(' ', '<space>')
        # Находим команды LaTeX (начинаются с \) или отдельные символы
        # Регулярное выражение: команда LaTeX (\alpha, \frac) ИЛИ любой символ
        tokens = re.findall(r'(\.+? |\\[a-zA-Z]+|.)', formula)
        # Убираем лишние пробелы и возвращаем непустые токены
        tokens = [t.strip().replace('<space>',' ') for t in tokens if t.strip()] # Возвращаем пробел
        return tokens

    def numericalize(self, formula):
        """Преобразует формулу (строку) в список индексов.

        Добавляет SOS и EOS токены.
        """
        tokenized_formula = self._basic_tokenizer(formula)
        numericalized = [SOS_IDX]
        for token in tokenized_formula:
            numericalized.append(self.stoi.get(token, UNK_IDX))
        numericalized.append(EOS_IDX)
        return numericalized

    def denumericalize(self, indices, remove_special_tokens=True):
        """Преобразует список индексов обратно в строку.

        Args:
            indices: Список или тензор индексов.
            remove_special_tokens: Удалить ли токены SOS, EOS, PAD из результата.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            char = self.itos.get(idx)
            if char:
                if remove_special_tokens and char in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                    continue
                chars.append(char)
            else:
                # Если индекс неизвестен, можно добавить UNK или пропустить
                if remove_special_tokens:
                     continue
                else:
                    chars.append(UNK_TOKEN)
        # Объединяем токены. В идеале нужно более умное объединение для LaTeX.
        return "".join(chars)

# --- Функция для DataLoader --- (Collate Function)

def collate_fn(batch):
    """Обрабатывает батч данных из Dataset.

    Принимает список кортежей (изображение, нумеризованная_формула).
    Дополняет формулы паддингом до максимальной длины в батче.

    Returns:
        Кортеж: (images_batch, padded_formulas_batch, lengths_batch)
    """
    # Разделяем изображения и формулы
    images = [item[0] for item in batch]
    formulas = [torch.tensor(item[1], dtype=torch.long) for item in batch]

    # Добавляем паддинг к формулам
    # pad_sequence ожидает список тензоров и добавляет паддинг
    # batch_first=True делает выходной тензор формы (batch_size, seq_len)
    padded_formulas = pad_sequence(formulas, batch_first=True, padding_value=PAD_IDX)

    # Собираем изображения в один тензор
    # Предполагается, что images - это уже тензоры нужного размера
    images_batch = torch.stack(images, dim=0)

    # Получаем реальные длины последовательностей (включая SOS и EOS)
    # Они нужны для pack_padded_sequence (если используется) или для маскирования потерь
    lengths = [len(f) for f in formulas]
    lengths_batch = torch.tensor(lengths, dtype=torch.long)

    return images_batch, padded_formulas, lengths_batch

# --- Пример использования --- (Можно раскомментировать для проверки)
# if __name__ == '__main__':
#     # Пример LaTeX формул
#     formulas_example = [
#         "y = x^{2}",
#         "\\frac{a+b}{c}",
#         "\\sin ( \\theta ) = 0.5"
#     ]
#
#     # 1. Создание и построение словаря
#     vocab = Vocabulary(freq_threshold=1)
#     vocab.build_vocabulary(formulas_example)
#
#     print("\nСловарь (stoi):")
#     print(vocab.stoi)
#     print("\nСловарь (itos):")
#     print(vocab.itos)
#     print(f"\nРазмер словаря: {len(vocab)}")
#
#     # 2. Нумеризация примера
#     formula_str = formulas_example[0]
#     numericalized = vocab.numericalize(formula_str)
#     print(f"\nФормула: '{formula_str}'")
#     print(f"Нумеризовано: {numericalized}")
#
#     # 3. Денумеризация примера
#     denumericalized = vocab.denumericalize(numericalized)
#     print(f"Денумеризовано: '{denumericalized}'")
#     denumericalized_no_special = vocab.denumericalize(numericalized, remove_special_tokens=True)
#     print(f"Денумеризовано (без спец): '{denumericalized_no_special}'")
#
#     # 4. Пример работы collate_fn
#     print("\nТест collate_fn:")
#     # Создаем фиктивные изображения и нумеризованные формулы разной длины
#     # Размеры изображений (C, H, W) должны быть одинаковыми перед collate_fn!
#     img_shape = (1, 64, 256)
#     img1 = torch.randn(img_shape)
#     form1 = vocab.numericalize("a=1")
#     img2 = torch.randn(img_shape)
#     form2 = vocab.numericalize("x^{2}+y^{2}=z^{2}")
#     img3 = torch.randn(img_shape)
#     form3 = vocab.numericalize("b")
#
#     batch_example = [(img1, form1), (img2, form2), (img3, form3)]
#
#     images_b, formulas_b, lengths_b = collate_fn(batch_example)
#
#     print(f"Форма батча изображений: {images_b.shape}")
#     print(f"Форма батча формул (с паддингом): {formulas_b.shape}")
#     print(f"Батч формул:\n{formulas_b}")
#     print(f"Реальные длины формул: {lengths_b.tolist()}")
#
#     # Проверяем паддинг
#     print(f"Индекс PAD токена: {PAD_IDX}")
