# Основной скрипт пайплайна

# import argparse
# from preprocessing.image_processing import preprocess_image
# from recognition.predict import predict # Предполагается, что predict принимает предобработанное изображение
# from postprocessing.formatter import format_output
# from solver.sympy_solver import solve_equation_sympy

# def run_pipeline(image_path, model_path):
#     # 1. Предобработка
#     processed_image = preprocess_image(image_path)
#     # 2. Распознавание
#     model_output = predict(processed_image, model_path) # Уточнить интерфейс predict
#     # 3. Форматирование
#     equation_str = format_output(model_output)
#     print(f"Распознанное уравнение: {equation_str}")
#     # 4. Решение
#     solution = solve_equation_sympy(equation_str)
#     print(f"Решение: {solution}")
#     return equation_str, solution

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Распознавание и решение рукописных уравнений.")
#     parser.add_argument("--image_path", type=str, required=True, help="Путь к изображению уравнения.")
#     parser.add_argument("--model_path", type=str, required=True, help="Путь к файлу обученной модели.")
#     args = parser.parse_args()

#     run_pipeline(args.image_path, args.model_path)

print("Файл для основного пайплайна") 