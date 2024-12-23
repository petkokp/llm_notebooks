import re

def find_numbers(x: str) -> list[str]:
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str,
                answer_delimiter: str = 'Отговорът е') -> str: # 'The answer is'
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''
