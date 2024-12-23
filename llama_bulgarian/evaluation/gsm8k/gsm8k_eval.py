from datasets import load_dataset
from preamble import PREAMBLE
from prompt import PROMPT
from maybe_remove_comma import maybe_remove_comma
from find_number import find_number
from predict import predict

GSM8K_BG_ID = "INSAIT-Institute/GSM8k-bgeval"

gsm8k = load_dataset(GSM8K_BG_ID, cache_dir='/tmp')
gsm8k_test = gsm8k['test']

all_correct = 0
all_responses = {}
short_responses = {}
idx = 0
correct = 0

TEMPLATE = """
Q: {question}
A:"""

for task_id, problem in enumerate(gsm8k_test):
  if task_id in all_responses: continue
  print(f"task_id {task_id}")
  full_prompt = (PREAMBLE +'\n\n' + PROMPT + '\n' +
                 TEMPLATE.format(question=problem['question']))
  short_prompt = PREAMBLE +'\n' + TEMPLATE.format(question=problem['question'])

  response = predict(full_prompt)

  all_responses[task_id] = response.split('\nQ:')[0]
  short_responses[task_id] = maybe_remove_comma(find_number(all_responses[task_id]))
  print(f"Short answer: {short_responses[task_id]}")
  try:
    correct += float(maybe_remove_comma(
        find_number(problem['answer']))) == float(short_responses[task_id])
  except:
    correct += maybe_remove_comma(
        find_number(problem['answer'])) == maybe_remove_comma(
            find_number(short_responses[task_id]))
  print('-'*40)
  print(f"Ground truth answer {problem['answer']}")
  print(f"Short ground truth answer {find_number(problem['answer'])}")
  print(f"Correct: {correct} out of {idx+1}")
  print("="*40)
  idx += 1