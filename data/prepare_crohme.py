import os
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm # Для отображения прогресса

# --- Функция парсинга InkML --- 

def extract_latex_from_inkml(inkml_file_path):
    """Извлекает LaTeX строку из аннотации 'truth' в InkML файле.

    Args:
        inkml_file_path (str): Путь к .inkml файлу.

    Returns:
        str or None: Извлеченная LaTeX строка или None, если не найдена или ошибка.
    """
    try:
        tree = ET.parse(inkml_file_path)
        root = tree.getroot()
        
        # InkML часто использует пространства имен. Нужно их учитывать.
        # Примерный поиск пространства имен (может потребоваться адаптация под конкретный файл)
        namespaces = {'ns': root.tag.split('}')[0].strip('{') if '}' in root.tag else ''}
        if namespaces['ns']: # Если пространство имен найдено
            truth_annotation = root.find(f".//ns:annotation[@type='truth']", namespaces)
            # Иногда правда может быть в другом теге, например, <annotationXML> для MathML
            if truth_annotation is None:
                 truth_annotation = root.find(f".//ns:annotationXML[@encoding='MathML-Content']", namespaces)
                 # TODO: Если нашли MathML, его нужно будет конвертировать в LaTeX!
                 # Пока просто вернем None или саму строку MathML с предупреждением
                 if truth_annotation is not None: 
                      print(f"Предупреждение: Найдена аннотация MathML, а не LaTeX в {inkml_file_path}")
                      # return truth_annotation.text.strip() # Вернет MathML
                      return None # Пока пропускаем MathML
        else:
            truth_annotation = root.find(".//annotation[@type='truth']")
            if truth_annotation is None:
                 truth_annotation = root.find(".//annotationXML[@encoding='MathML-Content']")
                 if truth_annotation is not None:
                      print(f"Предупреждение: Найдена аннотация MathML, а не LaTeX в {inkml_file_path}")
                      # return truth_annotation.text.strip()
                      return None

        if truth_annotation is not None and truth_annotation.text:
            # Убираем возможные лишние пробелы и переносы строк
            return truth_annotation.text.strip()
        else:
            # print(f"Предупреждение: Аннотация 'truth' не найдена в {inkml_file_path}") # Убрано, чтобы не засорять вывод
            return None
            
    except ET.ParseError as e:
        print(f"Ошибка парсинга XML в файле {inkml_file_path}: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при обработке файла {inkml_file_path}: {e}")
        return None

# --- Основная функция обработки директории --- 

def process_directory(base_dir, inkml_subdir, img_subdir, output_file):
    """Обрабатывает директорию с данными, извлекает пути и LaTeX, сохраняет в файл.

    Args:
        base_dir (str): Базовая директория датасета (например, 'TC11_CROHME23').
        inkml_subdir (str): Поддиректория с InkML файлами (например, 'INKML/train').
        img_subdir (str): Поддиректория с PNG файлами (например, 'IMG/train').
        output_file (str): Путь к выходному файлу (например, 'data/train_labels.txt').
    """
    print(f"Обработка директории: {os.path.join(base_dir, inkml_subdir)} -> {output_file}")
    count = 0
    skipped_no_truth = 0
    skipped_no_png = 0
    inkml_root = os.path.join(base_dir, inkml_subdir)
    img_root = os.path.join(base_dir, img_subdir)
    
    # Создаем директорию для выходного файла, если ее нет
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Рекурсивный обход директории inkml_root
        for root, _, files in os.walk(inkml_root):
            for filename in tqdm(files, desc=f"Файлы в {os.path.basename(root)}", leave=False):
                if filename.endswith('.inkml'):
                    inkml_path = os.path.join(root, filename)
                    
                    # Извлекаем LaTeX
                    latex_str = extract_latex_from_inkml(inkml_path)
                    if latex_str is None:
                        skipped_no_truth += 1
                        continue # Пропускаем файл, если нет аннотации truth

                    # Формируем путь к соответствующему PNG
                    # Получаем относительный путь inkml от корня inkml_subdir
                    relative_inkml_path = os.path.relpath(inkml_path, inkml_root)
                    # Заменяем расширение и формируем путь к PNG
                    png_filename = os.path.splitext(filename)[0] + '.png'
                    relative_png_dir = os.path.dirname(relative_inkml_path)
                    # Путь к PNG относительно БАЗОВОЙ директории датасета
                    relative_png_path = os.path.join(img_subdir, relative_png_dir, png_filename)
                    # Нормализуем разделители пути для единообразия
                    relative_png_path = relative_png_path.replace('\\', '/') 
                    
                    # Составляем полный путь для проверки существования
                    full_png_path = os.path.join(base_dir, relative_png_path)

                    # Проверяем, существует ли PNG файл
                    if not os.path.exists(full_png_path):
                        # print(f"Предупреждение: PNG файл не найден для {inkml_path} (ожидался {full_png_path})")
                        skipped_no_png += 1
                        continue # Пропускаем, если нет PNG

                    # Записываем в выходной файл (относительный путь PNG <TAB> LaTeX)
                    outfile.write(f"{relative_png_path}\t{latex_str}\n")
                    count += 1

    print(f"Обработка завершена. Записано {count} пар (изображение, формула).")
    if skipped_no_truth > 0:
        print(f"Пропущено из-за отсутствия аннотации 'truth': {skipped_no_truth}")
    if skipped_no_png > 0:
        print(f"Пропущено из-за отсутствия PNG файла: {skipped_no_png}")
    print("---")


# --- Главная часть скрипта --- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Подготовка данных CROHME: создание файлов разметки.")
    parser.add_argument("--base_dir", type=str, default="TC11_CROHME23", 
                        help="Путь к базовой директории датасета CROHME.")
    parser.add_argument("--output_dir", type=str, default="data", 
                        help="Директория для сохранения выходных файлов (_labels.txt).")
    args = parser.parse_args()

    # Определяем поддиректории и выходные файлы
    sets_to_process = {
        'train': (os.path.join('INKML', 'train'), os.path.join('IMG', 'train'), os.path.join(args.output_dir, 'train_labels.txt')),
        'val':   (os.path.join('INKML', 'val'), os.path.join('IMG', 'val'), os.path.join(args.output_dir, 'val_labels.txt')),
        'test':  (os.path.join('INKML', 'test'), os.path.join('IMG', 'test'), os.path.join(args.output_dir, 'test_labels.txt'))
    }

    # Обрабатываем каждую выборку
    for set_name, (inkml_subdir, img_subdir, output_file) in sets_to_process.items():
        process_directory(args.base_dir, inkml_subdir, img_subdir, output_file)

    print("Все выборки обработаны.")

print("Функция extract_latex_from_inkml определена.") 