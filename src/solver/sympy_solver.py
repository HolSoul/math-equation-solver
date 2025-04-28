# Реализация решателя уравнений с использованием SymPy

from sympy import sympify, solve, symbols, Eq, S
from sympy.core.sympify import SympifyError
from typing import Any

def solve_equation_sympy(equation_str: str) -> Any:
    """Решает математическое уравнение, представленное в виде строки, используя SymPy.

    Предполагается, что строка содержит знак равенства '='.
    Степень обозначается как '**'.

    Args:
        equation_str: Строка с уравнением (например, "y**4 + y + 1 = 0").

    Returns:
        Решение(я), найденные SymPy (обычно список), или строку с описанием ошибки.
    """
    if '=' not in equation_str:
        return "Ошибка: Уравнение должно содержать знак '='."

    try:
        # Разделяем уравнение на левую и правую части
        lhs_str, rhs_str = equation_str.split('=', 1)
        lhs = sympify(lhs_str)
        rhs = sympify(rhs_str)

        # Собираем все символы (переменные) из обеих частей
        variables = lhs.free_symbols.union(rhs.free_symbols)

        if not variables:
            # Случай без переменных
            if lhs.equals(rhs):
                return "Уравнение является тождеством (верно для любых переменных)."
            else:
                return "Уравнение не содержит переменных и не является тождеством."
        else:
            # Случай с переменными
            # Проверяем на символьное тождество или противоречие ДО вызова solve
            equation_diff = lhs - rhs
            # Пытаемся упростить разность
            simplified_diff = equation_diff.simplify()

            if simplified_diff.is_zero:
                 return "Уравнение является тождеством (верно для всех значений переменных)."
            # Некоторые противоречия могут не упроститься до константы, но solve вернет []
            # if simplified_diff.is_constant() and not simplified_diff.is_zero:
            #     return "Уравнение является противоречием (неверно для всех значений переменных)." # solve вернет []

            # Решаем уравнение lhs - rhs = 0 относительно всех переменных
            solution = solve(equation_diff, *variables)

            # Обработка пустого списка решений от solve
            if not solution:
                # Проверяем, было ли это противоречие (упрощенная разность не равна нулю)
                if simplified_diff.is_constant() and not simplified_diff.is_zero:
                     return "Уравнение является противоречием."
                else:
                    # Если solve вернул пустой список, но это не явное противоречие,
                    # возможно, решений нет или они слишком сложны
                    return "SymPy не нашел явных решений."

            return solution

    except (SympifyError, TypeError, ValueError, NotImplementedError) as e:
        return f"Ошибка разбора или решения уравнения: {e}"

# Пример использования
if __name__ == '__main__':
    examples = [
        "x + 5 = 10",
        "y**2 - 4 = 0",
        "z * 3 = z + 1",
        "a**2 + 1 = 0", # Комплексные решения
        "y**4 + y + 1 = 0", # Пример из CROHME
        "sin(x) = 0.5", # Тригонометрия
        "5 = 5", # Тождество
        "1 = 0", # Нет переменных, не тождество
        "x + y = 1", # Несколько переменных (solve вернет решение для одной через другую)
        "invalid-equation", # Ошибка парсинга
        "x+", # Ошибка парсинга
        "x = " # Ошибка парсинга
    ]

    for eq_str in examples:
        print(f"Уравнение: {eq_str}")
        result = solve_equation_sympy(eq_str)
        print(f"Решение:   {result}")
        print("---")

print("Файл для решателя на базе SymPy") 