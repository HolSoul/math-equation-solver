import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, input_channels=1, output_features=512):
        """CNN Encoder для извлечения признаков из изображения.

        Args:
            input_channels: Количество входных каналов изображения (1 для серого).
            output_features: Размерность выходных признаков (глубина карты признаков).
        """
        super(EncoderCNN, self).__init__()
        self.output_features = output_features

        # Определяем слои CNN
        self.cnn_layers = nn.Sequential(
            # Блок 1
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Уменьшает H, W в 2 раза

            # Блок 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Уменьшает H, W еще в 2 раза

            # Блок 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)), # Можно уменьшать только высоту

            # Блок 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Уменьшает H еще в 2 раза, W остается

            # Блок 5
            nn.Conv2d(256, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True)
            # Финальный MaxPool не обязателен, зависит от желаемого размера карты признаков
        )

        # print(f"Инициализирован EncoderCNN с {output_features} выходными признаками.") # Можно закомментировать

    def forward(self, x):
        """Прямой проход через CNN.

        Args:
            x: Входной тензор изображения (Batch, Channels, Height, Width).
               Ожидается, что Channels = input_channels (обычно 1).

        Returns:
            Тензор карты признаков (Batch, Features, Height_out, Width_out).
            Features = output_features.
            Height_out и Width_out зависят от начальных размеров и слоев пулинга.
        """
        features = self.cnn_layers(x)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """Механизм внимания (Bahdanau-style).

        Args:
            encoder_dim: Размерность признаков энкодера (output_features CNN).
            decoder_dim: Размерность скрытого состояния декодера (RNN hidden size).
            attention_dim: Размерность внутреннего слоя внимания.
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Линейный слой для признаков энкодера
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Линейный слой для скрытого состояния декодера
        self.full_att = nn.Linear(attention_dim, 1) # Линейный слой для вычисления "энергии" внимания
        self.relu = nn.ReLU() # Можно использовать tanh, как в оригинальной статье Bahdanau
        self.softmax = nn.Softmax(dim=1) # Softmax по "пикселям" карты признаков
        print("Инициализирован Attention (Bahdanau-style)")

    def forward(self, encoder_out, decoder_hidden):
        """Вычисление весов внимания и контекстного вектора.

        Args:
            encoder_out: Выход энкодера (Batch, Num_Pixels, Encoder_Dim).
                         Num_Pixels = Height_out * Width_out
            decoder_hidden: Скрытое состояние декодера (Batch, Decoder_Dim).

        Returns:
            context_vector: Контекстный вектор (Batch, Encoder_Dim).
            attention_weights: Веса внимания (Batch, Num_Pixels).
        """
        # Проекция признаков энкодера: (Batch, Num_Pixels, Encoder_Dim) -> (Batch, Num_Pixels, Attention_Dim)
        att1 = self.encoder_att(encoder_out)
        # Проекция скрытого состояния декодера: (Batch, Decoder_Dim) -> (Batch, Attention_Dim)
        att2 = self.decoder_att(decoder_hidden)

        # Сложение проекций с использованием broadcasting
        # Добавляем измерение Num_Pixels к att2: (Batch, Attention_Dim) -> (Batch, 1, Attention_Dim)
        # Затем складываем с att1: (Batch, Num_Pixels, Attention_Dim)
        # Применяем ReLU (или tanh)
        # Наконец, применяем full_att: (Batch, Num_Pixels, Attention_Dim) -> (Batch, Num_Pixels, 1)
        # Убираем последнюю размерность: -> (Batch, Num_Pixels)
        # Это "энергия" или "score" внимания для каждого пикселя
        attention_energy = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)

        # Применяем softmax для получения весов внимания
        # (Batch, Num_Pixels) -> (Batch, Num_Pixels), сумма по dim=1 равна 1
        alpha = self.softmax(attention_energy)

        # Вычисляем контекстный вектор как взвешенную сумму признаков энкодера
        # encoder_out: (Batch, Num_Pixels, Encoder_Dim)
        # alpha.unsqueeze(2): (Batch, Num_Pixels, 1)
        # Перемножаем поэлементно и суммируем по измерению Num_Pixels (dim=1)
        # Результат: (Batch, Encoder_Dim)
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context_vector, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, attention_dim, dropout=0.5):
        """RNN Decoder с механизмом внимания и LSTM.

        Args:
            embed_dim: Размерность эмбеддингов для входных символов декодера.
            decoder_dim: Размерность скрытого состояния LSTM.
            vocab_size: Размер словаря (количество уникальных символов + спец. токены).
            encoder_dim: Размерность признаков энкодера.
            attention_dim: Размерность слоя внимания.
            dropout: Вероятность dropout.
        """
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim # Сохраняем embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Используем LSTMCell
        # Вход LSTMCell = конкатенация эмбеддинга текущего токена и контекстного вектора
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim) # Для инициализации скрытого состояния (h)
        self.init_c = nn.Linear(encoder_dim, decoder_dim) # Для инициализации состояния ячейки (c)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim) # Слой для sigmoid gate (адаптивное внимание)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size) # Финальный классификационный слой
        self.dropout = nn.Dropout(p=dropout)

        print(f"Инициализирован DecoderRNN с LSTMCell (decoder_dim={decoder_dim})")

    def init_hidden_state(self, encoder_out):
        """Инициализация скрытого состояния (h) и состояния ячейки (c) LSTM.

        Используется среднее значение признаков энкодера по всем пикселям.

        Args:
            encoder_out: Выход энкодера (Batch, Num_Pixels, Encoder_Dim).

        Returns:
            tuple: Кортеж (h, c) с начальными состояниями.
                   h: (Batch, Decoder_Dim)
                   c: (Batch, Decoder_Dim)
        """
        # Усредняем признаки энкодера по измерению Num_Pixels (dim=1)
        mean_encoder_out = encoder_out.mean(dim=1)
        # Инициализируем h и c через линейные слои
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """Прямой проход через декодер с использованием teacher forcing.

        Args:
            encoder_out: Выход энкодера (Batch, Num_Pixels, Encoder_Dim).
            encoded_captions: Закодированные целевые последовательности (Batch, Max_Len).
                               Включают токен <start>, но не <end>.
            caption_lengths: Реальные длины последовательностей (Batch), не включая <start>.

        Returns:
            predictions: Предсказания логитов для каждого шага (Batch, Max_Len, Vocab_Size).
            alphas: Веса внимания для каждого шага (Batch, Max_Len, Num_Pixels).
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Получаем эмбеддинги для входных подписей
        # (Batch, Max_Len) -> (Batch, Max_Len, Embed_Dim)
        embeddings = self.embedding(encoded_captions)

        # Инициализируем LSTM состояние (h, c)
        h, c = self.init_hidden_state(encoder_out)

        # Определяем максимальную длину декодирования на основе длин подписей
        # Уменьшаем длины на 1, так как мы не декодируем токен <end>
        # (но нам нужны реальные длины для создания маски)
        decode_lengths = [l - 1 for l in caption_lengths]
        max_decode_length = max(decode_lengths)

        # Тензоры для хранения предсказаний и весов внимания
        predictions = torch.zeros(batch_size, max_decode_length, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_length, encoder_out.size(1)).to(encoder_out.device)

        # Цикл по временным шагам декодирования
        for t in range(max_decode_length):
            # Определяем батчи, которые еще нужно декодировать на этом шаге
            batch_size_t = sum([l > t for l in decode_lengths])
            # Выбираем активные батчи для энкодера и состояний LSTM
            active_encoder_out = encoder_out[:batch_size_t]
            active_h = h[:batch_size_t]
            active_c = c[:batch_size_t]

            # Вычисляем контекстный вектор и веса внимания
            # attention_vector: (batch_size_t, encoder_dim)
            # alpha: (batch_size_t, num_pixels)
            attention_vector, alpha = self.attention(active_encoder_out, active_h)

            # Адаптивное внимание (опционально, можно пропустить)
            # gate = self.sigmoid(self.f_beta(active_h))
            # attention_vector = gate * attention_vector

            # Получаем эмбеддинг для текущего временного шага
            # embeddings: (Batch, Max_Len, Embed_Dim) -> (batch_size_t, Embed_Dim)
            current_embedding = embeddings[:batch_size_t, t, :]

            # Конкатенируем эмбеддинг и контекстный вектор
            # lstm_input: (batch_size_t, embed_dim + encoder_dim)
            lstm_input = torch.cat((current_embedding, attention_vector), dim=1)

            # Выполняем шаг LSTM
            # h: (batch_size_t, decoder_dim)
            # c: (batch_size_t, decoder_dim)
            h_next, c_next = self.decode_step(lstm_input, (active_h, active_c))

            # Применяем dropout к скрытому состоянию
            preds_t = self.dropout(h_next)
            # Прогоняем через полносвязный слой для получения логитов
            # preds_t: (batch_size_t, vocab_size)
            preds_t = self.fc(preds_t)

            # Сохраняем предсказания и веса внимания
            predictions[:batch_size_t, t, :] = preds_t
            alphas[:batch_size_t, t, :] = alpha

            # Обновляем состояния LSTM для следующего шага
            h = h_next
            c = c_next

        return predictions, alphas

class RecognitionModel(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_output_features=512, attention_dim=512, dropout=0.5):
        """Полная модель распознавания, объединяющая Encoder и Decoder.

        Args:
            embed_dim: Размерность эмбеддингов декодера.
            decoder_dim: Размерность скрытого состояния декодера.
            vocab_size: Размер словаря.
            encoder_output_features: Размерность выхода CNN энкодера.
            attention_dim: Размерность слоя внимания.
            dropout: Вероятность dropout в декодере.
        """
        super(RecognitionModel, self).__init__()
        self.encoder = EncoderCNN(output_features=encoder_output_features)
        self.decoder = DecoderRNN(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_output_features,
            attention_dim=attention_dim,
            dropout=dropout
        )
        # print("Инициализирована RecognitionModel")

    def forward(self, images, encoded_captions, caption_lengths):
        """Прямой проход через всю модель.

        Args:
            images: Входные изображения (Batch, Channels, Height, Width).
            encoded_captions: Закодированные целевые последовательности (Batch, Max_Len).
            caption_lengths: Реальные длины последовательностей (Batch).

        Returns:
            predictions: Предсказания декодера (Batch, Max_Len, Vocab_Size).
        """
        features = self.encoder(images)
        b, c, h, w = features.size()
        features = features.view(b, c, -1)
        features = features.permute(0, 2, 1)

        # Передаем features (выход энкодера) в декодер
        predictions, alphas = self.decoder(features, encoded_captions, caption_lengths)
        # Пока возвращаем только предсказания, но можно и alphas для анализа
        return predictions

# --- Параметры (примерные) ---
# Нужно будет определить на основе данных
EMBED_DIM = 256
DECODER_DIM = 512
VOCAB_SIZE = 100 # Примерное количество символов + спец. токены (PAD, SOS, EOS)
ATTENTION_DIM = 512
ENCODER_FEATURES = 512
DROPOUT = 0.5

# --- Пример создания модели ---
# model = RecognitionModel(
#     embed_dim=EMBED_DIM,
#     decoder_dim=DECODER_DIM,
#     vocab_size=VOCAB_SIZE,
#     encoder_output_features=ENCODER_FEATURES,
#     attention_dim=ATTENTION_DIM,
#     dropout=DROPOUT
# )
# print(model)

# --- Пример фиктивного входа ---
# batch_size = 4
# img_height = 64
# img_width = 256 # Примерная ширина после ресайза
# max_caption_len = 50

# dummy_images = torch.randn(batch_size, 1, img_height, img_width)
# dummy_captions = torch.randint(0, VOCAB_SIZE, (batch_size, max_caption_len))
# dummy_lengths = torch.full((batch_size,), max_caption_len, dtype=torch.long)

# TODO: Раскомментировать и запустить, когда будут реализованы forward методы
# try:
#     # predictions = model(dummy_images, dummy_captions, dummy_lengths)
#     # print("Форма выхода модели:", predictions.shape) # Ожидаем (Batch, Max_Len, Vocab_Size)
#     pass
# except NotImplementedError as e:
#     print(f"Не удалось выполнить forward: {e}")

print("Каркас модели распознавания создан.") 