

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import json # Для сохранения словаря

# Импортируем наши компоненты
from model import RecognitionModel
from dataset import LatexDataset
from utils import Vocabulary, collate_fn, PAD_IDX

# TODO: Импортировать функцию для загрузки данных (image_paths, formulas) из файла разметки
# from data_loader import load_data_from_file

def train(args):
    """Основная функция обучения модели."""

    # --- Настройка --- 
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Создаем директорию для сохранения результатов (модели, словаря)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Загрузка данных и построение словаря --- 
    # TODO: Реализовать загрузку реальных данных!
    # Это ЗАГЛУШКА - нужно заменить на реальную загрузку путей и формул
    print("Загрузка данных (ЗАГЛУШКА)...")
    # train_image_paths, train_formulas = load_data_from_file(args.train_data_path)
    # val_image_paths, val_formulas = load_data_from_file(args.val_data_path)
    # Примерные данные для заглушки:
    train_formulas = ["a=1", "x^{2}+y^{2}=z^{2}", "b", "\sin(x)=0"] 
    train_image_paths = ["dummy_path1.png", "dummy_path2.png", "dummy_path3.png", "dummy_path4.png"] # Пути должны быть реальными!
    val_formulas = ["c=d+e", "\log(z)"]
    val_image_paths = ["dummy_path5.png", "dummy_path6.png"]

    # Создание или загрузка словаря
    vocab_path = os.path.join(args.checkpoint_dir, 'vocab.json')
    if os.path.exists(vocab_path) and not args.rebuild_vocab:
        print(f"Загрузка словаря из {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        vocab = Vocabulary(freq_threshold=args.vocab_threshold)
        vocab.stoi = vocab_data['stoi']
        vocab.itos = {int(k): v for k, v in vocab_data['itos'].items()} # Ключи json - строки
        print(f"Словарь загружен. Размер: {len(vocab)}")
    else:
        print("Построение словаря...")
        vocab = Vocabulary(freq_threshold=args.vocab_threshold)
        # Строим словарь ТОЛЬКО по обучающим данным!
        vocab.build_vocabulary(train_formulas)
        print(f"Сохранение словаря в {vocab_path}")
        # Сохраняем словарь для будущего использования
        vocab_data = {'stoi': vocab.stoi, 'itos': vocab.itos}
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    VOCAB_SIZE = len(vocab)
    print(f"Итоговый размер словаря: {VOCAB_SIZE}")

    # --- Создание Dataset и DataLoader --- 
    # TODO: Убедиться, что пути к изображениям в заглушке существуют или заменить их
    # TODO: Обработка ошибок в LatexDataset (возвращает None)
    print("Создание Dataset и DataLoader...")
    train_dataset = LatexDataset(train_image_paths, train_formulas, vocab, target_height=args.img_height)
    val_dataset = LatexDataset(val_image_paths, val_formulas, vocab, target_height=args.img_height)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True # Может ускорить передачу данных на GPU
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Валидацию не перемешиваем
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print("DataLoader'ы созданы.")

    # --- Инициализация модели, потерь и оптимизатора --- 
    print("Инициализация модели...")
    model = RecognitionModel(
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=VOCAB_SIZE,
        encoder_output_features=args.encoder_features,
        attention_dim=args.attention_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Игнорируем паддинг при расчете потерь
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # TODO: Добавить планировщик скорости обучения (lr_scheduler), если нужно

    # --- Цикл обучения --- 
    print("Начало обучения...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train() # Переводим модель в режим обучения
        total_train_loss = 0

        for i, batch in enumerate(train_loader):
            if batch is None: # Пропускаем, если collate_fn не смог собрать батч
                print(f"Пропущен None батч на шаге {i} обучения.")
                continue
                
            images, formulas, lengths = batch
            images = images.to(device)
            formulas = formulas.to(device)
            # lengths не обязательно переносить на device, если не используются в модели явно

            # --- Прямой проход --- 
            # Вход для декодера: формулы без последнего токена (<eos>)
            # Цель для декодера: формулы без первого токена (<sos>)
            # caption_lengths передаются в декодер для цикла
            # Нужно передавать реальные длины (без <sos>)
            decoder_input = formulas[:, :-1]
            decoder_target = formulas[:, 1:]
            # Длины для цикла декодера (реальная длина - 1, так как <eos> не предсказываем)
            # Важно: длины должны быть на CPU для использования в цикле DecoderRNN
            decode_lengths = (lengths - 1).tolist()
            
            predictions, alphas = model(images, decoder_input, decode_lengths)
            
            # --- Расчет потерь --- 
            # predictions: (batch_size, max_decode_len, vocab_size)
            # decoder_target: (batch_size, max_decode_len)
            # Нужно изменить форму для CrossEntropyLoss:
            # predictions -> (batch_size * max_decode_len, vocab_size)
            # decoder_target -> (batch_size * max_decode_len)
            batch_size, seq_len, vocab_size_out = predictions.size()
            loss = criterion(predictions.reshape(-1, vocab_size_out), decoder_target.reshape(-1))
            total_train_loss += loss.item()

            # --- Обратный проход и оптимизация --- 
            optimizer.zero_grad()
            loss.backward()
            # TODO: Добавить обрезку градиента (gradient clipping), если нужно
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # Логирование
            if (i + 1) % args.log_interval == 0:
                print(f"Эпоха [{epoch+1}/{args.epochs}], Шаг [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Валидация --- 
        model.eval() # Переводим модель в режим оценки
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue # Пропуск
                images, formulas, lengths = batch
                images = images.to(device)
                formulas = formulas.to(device)
                
                decoder_input = formulas[:, :-1]
                decoder_target = formulas[:, 1:]
                decode_lengths = (lengths - 1).tolist()

                predictions, _ = model(images, decoder_input, decode_lengths)
                
                batch_size, seq_len, vocab_size_out = predictions.size()
                loss = criterion(predictions.reshape(-1, vocab_size_out), decoder_target.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"\nЭпоха [{epoch+1}/{args.epochs}] Завершена за {epoch_time:.2f} сек")
        print(f"Средняя Train Loss: {avg_train_loss:.4f}, Средняя Val Loss: {avg_val_loss:.4f}")

        # --- Сохранение модели --- 
        if avg_val_loss < best_val_loss:
            print(f"Val Loss улучшился ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Сохранение модели...")
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_stoi': vocab.stoi, # Сохраняем словарь вместе с моделью
                'vocab_itos': vocab.itos,
                # Сохраняем гиперпараметры для воспроизводимости
                'args': vars(args) 
            }, save_path)
            print(f"Модель сохранена в {save_path}")
        else:
            print("Val Loss не улучшился.")
        print("-"*30)

    total_training_time = time.time() - start_time
    print(f"Обучение завершено за {total_training_time // 60:.0f} мин {total_training_time % 60:.0f} сек")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели распознавания LaTeX формул")

    # Аргументы данных
    parser.add_argument("--train_data_path", type=str, required=True, help="Путь к файлу разметки обучающих данных")
    parser.add_argument("--val_data_path", type=str, required=True, help="Путь к файлу разметки валидационных данных")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Директория для сохранения моделей и словаря")
    parser.add_argument("--vocab_threshold", type=int, default=1, help="Минимальная частота для добавления токена в словарь")
    parser.add_argument("--rebuild_vocab", action='store_true', help="Перестроить словарь, даже если файл существует")
    parser.add_argument("--img_height", type=int, default=64, help="Целевая высота изображения")

    # Аргументы модели
    parser.add_argument("--embed_dim", type=int, default=256, help="Размерность эмбеддингов")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Размерность скрытого состояния декодера (LSTM)")
    parser.add_argument("--encoder_features", type=int, default=512, help="Количество признаков на выходе энкодера")
    parser.add_argument("--attention_dim", type=int, default=512, help="Размерность слоя внимания")
    parser.add_argument("--dropout", type=float, default=0.5, help="Вероятность dropout")

    # Аргументы обучения
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Скорость обучения (learning rate)")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Максимальная норма градиента для обрезки (0 для отключения)")
    parser.add_argument("--num_workers", type=int, default=0, help="Количество воркеров для DataLoader (0 для Windows)")
    parser.add_argument("--log_interval", type=int, default=100, help="Как часто выводить лог обучения (каждые N батчей)")
    
    args = parser.parse_args()
    
    # Выводим аргументы для информации
    print("\n--- Параметры Запуска ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-"*25)

    # Запускаем обучение
    train(args) 