import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
import argparse
import time
from tqdm import tqdm

# Импортируем наши компоненты
from model import RecognitionModel
from dataset import LatexDataset
from utils import Vocabulary, collate_fn, PAD_IDX

# TODO: Импортировать функцию для загрузки данных (image_paths, formulas) из файла разметки
# from data_loader import load_data_from_file

# --- Вспомогательная функция для загрузки файла разметки --- 
def load_label_file(label_file_path, image_base_dir):
    """Загружает файл разметки и возвращает списки путей к изображениям и формул.
    
    Args:
        label_file_path (str): Путь к файлу _labels.txt (например, 'data/train_labels.txt').
        image_base_dir (str): Базовый путь к директории с изображениями (например, 'TC11_CROHME23').

    Returns:
        tuple: Кортеж из двух списков (image_paths, formulas).
               image_paths содержит полные пути к файлам изображений.
    """
    image_paths = []
    formulas = []
    print(f"Загрузка файла разметки: {label_file_path}")
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Чтение строк разметки"):
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    relative_img_path, formula = parts
                    # Формируем полный путь к изображению
                    full_img_path = os.path.join(image_base_dir, relative_img_path)
                    # Добавляем только если изображение существует (доп. проверка)
                    if os.path.exists(full_img_path):
                        image_paths.append(full_img_path)
                        formulas.append(formula)
                    else:
                         print(f"Предупреждение: Изображение не найдено по пути: {full_img_path}, строка пропущена.")
                else:
                    print(f"Предупреждение: Некорректный формат строки в {label_file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Ошибка: Файл разметки не найден: {label_file_path}")
        return [], [] # Возвращаем пустые списки
    except Exception as e:
        print(f"Ошибка при чтении файла {label_file_path}: {e}")
        return [], [] # Возвращаем пустые списки
        
    print(f"Загружено {len(image_paths)} пар (изображение, формула) из {label_file_path}")
    return image_paths, formulas

def train(args):
    """Основная функция обучения модели."""
    # --- Настройка --- 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Используемое устройство: {device}")

    # Создаем директорию для сохранения результатов
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Результаты будут сохранены в: {args.output_dir}")

    # --- Загрузка данных --- 
    # Формируем пути к файлам разметки
    train_label_file = os.path.join(args.data_path, 'train_labels.txt')
    val_label_file = os.path.join(args.data_path, 'val_labels.txt')

    # Загружаем данные с помощью новой функции
    train_image_paths, train_formulas = load_label_file(train_label_file, args.image_path)
    val_image_paths, val_formulas = load_label_file(val_label_file, args.image_path)

    if not train_image_paths or not val_image_paths:
        print("Ошибка: Не удалось загрузить данные для обучения или валидации. Проверьте пути и файлы разметки.")
        return

    # --- Словарь --- 
    vocab_file_path = os.path.join(args.output_dir, args.vocab_file)
    if args.load_vocab and os.path.exists(vocab_file_path):
        print(f"Загрузка существующего словаря из {vocab_file_path}")
        with open(vocab_file_path, 'r') as f:
            word2idx = json.load(f)
        vocab = Vocabulary(word2idx=word2idx)
    else:
        print("Построение словаря по обучающим формулам...")
        vocab = Vocabulary()
        # Используем реальные формулы для построения словаря
        vocab.build_vocabulary(train_formulas)
        print(f"Сохранение словаря в {vocab_file_path}")
        with open(vocab_file_path, 'w') as f:
            json.dump(vocab.stoi, f, ensure_ascii=False, indent=4)
            
    vocab_size = len(vocab)
    print(f"Размер словаря: {vocab_size}")
    pad_token_key = '<pad>'
    if pad_token_key not in vocab.stoi:
        print(f"ОШИБКА: Токен '{pad_token_key}' не найден в словаре! Ключи словаря: {list(vocab.stoi.keys())}")
        raise ValueError(f"Критическая ошибка: Токен '{pad_token_key}' отсутствует в словаре.")
    pad_idx = vocab.stoi[pad_token_key]

    # --- Dataset и DataLoader --- 
    print("Создание Dataset и DataLoader...")
    # TODO: Добавить трансформации/аугментации для обучающего набора
    train_dataset = LatexDataset(image_paths=train_image_paths, 
                               formulas=train_formulas, 
                               vocab=vocab, 
                               target_height=args.img_height)
                               
    val_dataset = LatexDataset(image_paths=val_image_paths, 
                             formulas=val_formulas, 
                             vocab=vocab, 
                             target_height=args.img_height)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers, 
                              collate_fn=lambda batch: collate_fn(batch, pad_idx), # Передаем pad_idx
                              pin_memory=True if device == torch.device("cuda") else False)
                              
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers,
                            collate_fn=lambda batch: collate_fn(batch, pad_idx), # Передаем pad_idx
                            pin_memory=True if device == torch.device("cuda") else False)
                            
    print(f"Количество батчей в train_loader: {len(train_loader)}")
    print(f"Количество батчей в val_loader: {len(val_loader)}")

    # --- Модель, функция потерь, оптимизатор --- 
    print("Инициализация модели...")
    
    # Инициализируем RecognitionModel, передавая гиперпараметры
    model = RecognitionModel(
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=vocab_size,
        encoder_output_features=args.encoder_dim, # Размерность выхода энкодера
        attention_dim=args.attention_dim,         # Размерность attention
        dropout=args.dropout
    ).to(device) # Перемещаем на нужное устройство
    
    print(f"Количество обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Функция потерь (игнорируем PAD токен)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # TODO: Добавить Learning Rate Scheduler (например, StepLR или ReduceLROnPlateau)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # --- Цикл обучения --- 
    print("Начало цикла обучения...")
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        
        # --- Фаза обучения ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{args.num_epochs} [Обучение]")
        for i, (imgs, formulas) in enumerate(train_pbar):
            imgs = imgs.to(device)
            formulas = formulas.to(device)

            optimizer.zero_grad()

            # Формулы для входа декодера: <SOS> token1 token2 ... tokenN
            # Формулы для таргета: token1 token2 ... tokenN <EOS>
            outputs = model(imgs, formulas[:, :-1]) # Убираем <EOS> для входа
            
            # outputs shape: (batch_size, seq_len, vocab_size)
            # formulas shape: (batch_size, seq_len)
            # Нужно изменить размерность для CrossEntropyLoss:
            # outputs -> (batch_size * seq_len, vocab_size)
            # formulas -> (batch_size * seq_len)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), formulas[:, 1:].reshape(-1)) # Убираем <SOS> для таргета
            
            loss.backward()
            # Опционально: обрезка градиентов для предотвращения взрыва
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            
            # Обновление прогресс-бара
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Фаза валидации ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{args.num_epochs} [Валидация]")
        with torch.no_grad():
            for i, (imgs, formulas) in enumerate(val_pbar):
                imgs = imgs.to(device)
                formulas = formulas.to(device)

                outputs = model(imgs, formulas[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), formulas[:, 1:].reshape(-1))
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"Эпоха {epoch+1}/{args.num_epochs} завершена. "
              f"Время: {epoch_time:.2f}s, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"*** Новая лучшая модель сохранена в {save_path} (Val Loss: {best_val_loss:.4f}) ***")
            
        # Шаг для LR scheduler, если он используется
        # scheduler.step()

    total_time = time.time() - start_time
    print(f"Обучение завершено за {total_time // 60:.0f}м {total_time % 60:.0f}с")
    print(f"Лучшая Val Loss: {best_val_loss:.4f}")

# --- Парсер аргументов и запуск --- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение модели распознавания LaTeX формул.")
    
    # Пути
    parser.add_argument('--data_path', type=str, default='data', help='Путь к директории с файлами _labels.txt')
    parser.add_argument('--image_path', type=str, default='TC11_CROHME23', help='Базовый путь к директории с изображениями')
    parser.add_argument('--output_dir', type=str, default='models/recognition_outputs', help='Директория для сохранения словаря, логов и модели')
    parser.add_argument('--vocab_file', type=str, default='vocab.json', help='Имя файла словаря внутри output_dir')
    parser.add_argument('--load_vocab', action='store_true', help='Загрузить существующий словарь, если он есть')

    # Параметры модели
    parser.add_argument('--img_height', type=int, default=96, help='Высота, к которой приводятся изображения')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Размерность выхода CNN энкодера (используется как encoder_output_features)')
    parser.add_argument('--embed_dim', type=int, default=256, help='Размерность эмбеддингов токенов')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Размерность скрытого состояния LSTM/GRU декодера')
    parser.add_argument('--attention_dim', type=int, default=512, help='Размерность внутреннего слоя внимания')
    parser.add_argument('--dropout', type=float, default=0.5, help='Вероятность dropout в декодере')

    # Параметры обучения
    parser.add_argument('--num_epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Скорость обучения')
    parser.add_argument('--num_workers', type=int, default=2, help='Количество потоков для загрузки данных')
    parser.add_argument('--cpu', action='store_true', help='Использовать CPU вместо CUDA, даже если CUDA доступна')

    args = parser.parse_args()
    print("Используемые аргументы:")
    print(json.dumps(vars(args), indent=4))
    
    train(args) 