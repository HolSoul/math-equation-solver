print("Файл тестов для решателя")

import unittest
import sys
import os
import math
from sympy import symbols, I, Float, S, pi

# Добавляем корневую директорию проекта в путь
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.solver.sympy_solver import solve_equation_sympy

class TestSolver(unittest.TestCase):

    def test_simple_linear_equation(self):
        self.assertEqual(solve_equation_sympy("x + 5 = 10"), [S(5)])
        self.assertEqual(solve_equation_sympy("2*z - 6 = 0"), [S(3)])
        self.assertEqual(solve_equation_sympy("3*y = y + 1"), [S(1)/2])

    def test_quadratic_equation_real_roots(self):
        solutions = solve_equation_sympy("x**2 - 4 = 0")
        self.assertIsInstance(solutions, list)
        self.assertCountEqual(solutions, [S(-2), S(2)])

    def test_quadratic_equation_complex_roots(self):
        solutions = solve_equation_sympy("a**2 + 1 = 0")
        self.assertIsInstance(solutions, list)
        self.assertCountEqual(solutions, [-I, I])

    def test_polynomial_degree_4(self):
        solutions = solve_equation_sympy("y**4 + y + 1 = 0")
        self.assertIsInstance(solutions, list)
        self.assertEqual(len(solutions), 4)
        self.assertFalse(isinstance(solutions[0], str))

    def test_trigonometric_equation(self):
        solutions = solve_equation_sympy("sin(x) = 0.5")
        self.assertIsInstance(solutions, list)
        expected_solutions_numeric = [math.asin(0.5), math.pi - math.asin(0.5)]
        self.assertEqual(len(solutions), 2)
        self.assertTrue(all(isinstance(s, Float) or isinstance(s, float) for s in solutions))
        solution_floats = sorted([float(s) for s in solutions])
        expected_floats = sorted(expected_solutions_numeric)
        self.assertAlmostEqual(solution_floats[0], expected_floats[0], places=7)
        self.assertAlmostEqual(solution_floats[1], expected_floats[1], places=7)

    def test_identity(self):
        result_const = solve_equation_sympy("5 = 5")
        self.assertEqual(result_const, "Уравнение является тождеством (верно для любых переменных).")
        result_symbolic = solve_equation_sympy("x + 1 = x + 1")
        self.assertEqual(result_symbolic, "Уравнение является тождеством (верно для всех значений переменных).")

    def test_contradiction(self):
        result_const = solve_equation_sympy("1 = 0")
        self.assertEqual(result_const, "Уравнение не содержит переменных и не является тождеством.")
        result_symbolic = solve_equation_sympy("x = x + 1")
        self.assertEqual(result_symbolic, "Уравнение является противоречием.")

    def test_multiple_variables(self):
        solutions = solve_equation_sympy("x + y = 1")
        self.assertIsInstance(solutions, list)
        self.assertTrue(len(solutions) > 0)
        self.assertIsInstance(solutions[0], tuple)
        self.assertEqual(len(solutions[0]), 2)

    def test_missing_equals(self):
        result = solve_equation_sympy("x + 5")
        self.assertEqual(result, "Ошибка: Уравнение должно содержать знак '='.")

    def test_syntax_error(self):
        # Проверка отсутствия знака равенства (обрабатывается до sympify)
        result_plus = solve_equation_sympy("x+")
        self.assertEqual(result_plus, "Ошибка: Уравнение должно содержать знак '='.")

        # Проверка ошибки парсинга из-за пустой правой части
        result_empty_rhs = solve_equation_sympy("x = ")
        self.assertIsInstance(result_empty_rhs, str)
        self.assertTrue(result_empty_rhs.startswith("Ошибка разбора или решения"))

        # Проверка настоящей синтаксической ошибки (незакрытая скобка)
        result_unmatched_paren = solve_equation_sympy("x = (1 + ")
        self.assertIsInstance(result_unmatched_paren, str)
        self.assertTrue(result_unmatched_paren.startswith("Ошибка разбора или решения"))

if __name__ == '__main__':
    unittest.main() 