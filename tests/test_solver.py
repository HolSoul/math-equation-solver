# Тесты для модуля решателя
# import unittest
# from src.solver.sympy_solver import solve_equation_sympy
# from sympy import Symbol, Eq, solve

# class TestSolver(unittest.TestCase):
#     def test_simple_equation(self):
#         solution = solve_equation_sympy("x+5=10")
#         # SymPy может возвращать решение в разном виде, нужно адаптировать проверку
#         # Например, если ожидаем словарь {Symbol('x'): 5/2}
#         # Или если solve возвращает список [5/2]
#         # self.assertEqual(solution, [5/2]) # Пример
#         pass

#     def test_no_variable(self):
#         solution = solve_equation_sympy("5+5=10")
#         self.assertEqual(solution, "Уравнение не содержит переменных")

#     def test_invalid_equation(self):
#         solution = solve_equation_sympy("x+=")
#         self.assertTrue("Ошибка решения" in solution)
#     # Другие тесты...

# if __name__ == '__main__':
#     unittest.main()

print("Файл тестов для решателя") 