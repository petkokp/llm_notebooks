import random

def generate_equation():
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)

    operators = ['+', '-', '*', '/']
    
    operator = random.choice(operators)
    
    if operator == '/':
        num2 = random.randint(1, 100)
        num1 = num2 * random.randint(1, 10)
    
    equation = f"{num1}{operator}{num2}"
    
    answer = eval(equation)
    
    return equation, answer

def create_simple_arithmetic_dataset(n: int):
    DATASET_PATH = "simple_arithmetic_dataset.txt"
    with open(DATASET_PATH, "w") as file:
        for _ in range(n):
            equation, answer = generate_equation()
            file.write(f"{equation}={answer}\n")
    return DATASET_PATH
