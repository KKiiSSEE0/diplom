# error_examples.py

EXAMPLES = [
    {
        "title": "IndexError",
        "code": [
            "def calculate_averages(matrix):",
            "    sums = []",
            "    for row in matrix:",
            "        s = 0",
            "        for val in row:",
            "            s += val",
            "        sums.append(s / len(row))",
            "    print(\"Промежуточные суммы:\", sums)",
            "    for i in range(len(sums) + 1):  # ОШИБКА ЗДЕСЬ!",
            "        print(f\"Среднее значение для строки {i}: {sums[i]:.2f}\")",
            "    print(\"Расчёт завершён.\")",
            "",
            "data = [",
            "    [10, 20, 30],",
            "    [15, 25, 35],",
            "    [20, 40, 60]",
            "]",
            "calculate_averages(data)"
        ],
        "error_line": 9,
        "error_type": "IndexError",
        "explanation": "Перебор диапазона len(sums) + 1 приводит к выходу за границы списка. Используйте range(len(sums))."
    },
{
        "title": "IndexError в большой функции анализа данных",
        "code": [
            "def process_data(data):",
            "    results = []",
            "    for i, block in enumerate(data):",
            "        total = 0",
            "        for value in block:",
            "            total += value",
            "        results.append(total)",
            "    print('Результаты подсчёта:', results)",
            "    print('Детальный вывод:')",
            "    for i in range(len(results) + 10):  # ОШИБКА: выход за пределы списка",
            "        print(f'Block {i}: {results[i]}')",
            "    print('Обработка завершена.')",
            "",
            "# Генерация тестовых данных",
            "data = []",
            "for i in range(10):",
            "    data.append([j for j in range(i, i+10)])",
            "",
            "# Дополнительные неиспользуемые функции (ради объёма):",
        ] +
        [f"def helper_func_{n}():\n    return {n}*{n}" for n in range(1, 121)] + [
            "",
            "# Запуск основного анализа",
            "process_data(data)"
        ],
        "error_line": 10,  # здесь ошибка (10-я строка в списке code)
        "error_type": "IndexError",
        "explanation": (
            "В цикле for i in range(len(results) + 10) происходит попытка обратиться к несуществующим элементам списка. "
            "Правильно будет использовать for i in range(len(results))."
        )
    },
    {
        "title": "Ошибка типа (TypeError)",
        "code": [
            "def multiply_elements(lst, factor):",
            "    result = []",
            "    for item in lst:",
            "        result.append(item * factor)",
            "    return result",
            "",
            "def process():",
            "    values = [2, 4, 6, 8, 10]",
            "    coef = \"3\"      # ОШИБКА ЗДЕСЬ: строка, а не число!",
            "    res = multiply_elements(values, coef)",
            "    print(\"Результат умножения:\", res)",
            "    print(\"Обработка завершена.\")",
            "",
            "process()"
        ],
        "error_line": 8,
        "error_type": "TypeError",
        "explanation": "В функцию передаётся строка, а не число. Нужно использовать int(coef) перед вызовом функции."
    },
    {
        "title": "Ошибка значения (ValueError)",
        "code": [
            "def analyze_temperatures(temp_list):",
            "    max_temp = -1000",
            "    min_temp = 1000",
            "    sum_temp = 0",
            "    count = 0",
            "    for temp in temp_list:",
            "        value = float(temp)",
            "        if value > max_temp:",
            "            max_temp = value",
            "        if value < min_temp:",
            "            min_temp = value",
            "        sum_temp += value",
            "        count += 1",
            "    avg = sum_temp / count",
            "    print(f\"Максимум: {max_temp}, минимум: {min_temp}, среднее: {avg:.1f}\")",
            "",
            "data = [\"22.4\", \"19.8\", \"21.0\", \"Ошибка!\", \"23.2\", \"18.9\"]  # \"Ошибка!\" — ОШИБКА ЗДЕСЬ",
            "analyze_temperatures(data)"
        ],
        "error_line": 6,
        "error_type": "ValueError",
        "explanation": "В списке есть значение, которое нельзя преобразовать в число. Нужно обработать исключение try-except."
    }
]