from tqdm import tqdm

def find_highest_ranked_child(df, parent_id, base_dataset_parent_field, base_dataset_rank_field):
      children = df[df[base_dataset_parent_field] == parent_id]
      if not children.empty:
          return children.loc[children[base_dataset_rank_field].idxmin()]
      return None

def create_prompts(dataset, tokenizer, base_dataset_rank_field, base_dataset_parent_field, base_dataset_id_field, base_dataset_text_field, base_dataset_role_field, instruction_prompt, chat_template):
    threads = []
    df = dataset.to_pandas()

    max_rank = df[base_dataset_rank_field].max()
    df[base_dataset_rank_field].fillna(max_rank + 1, inplace=True)

    root_messages = df[df[base_dataset_parent_field].isna()]

    with tqdm(total=len(root_messages)) as pbar:
        for _, root_message in root_messages.iterrows():
            if root_message[base_dataset_text_field] is None:
                continue
            thread = [
                {
                    'content': instruction_prompt,
                    'role': 'system'
                },
                {
                    'content': root_message[base_dataset_text_field],
                    'role': 'user'
                }
            ]
            next_message = find_highest_ranked_child(df, root_message[base_dataset_id_field], base_dataset_parent_field, base_dataset_rank_field)
        
            while next_message is not None:
                role = next_message[base_dataset_role_field]
                if role == 'prompter':
                    role = 'user'
                thread.append({
                    'content': next_message[base_dataset_text_field],
                    'role': role
                })
                next_message = find_highest_ranked_child(df, next_message[base_dataset_id_field], base_dataset_parent_field, base_dataset_rank_field)

            try:
                threads.append({'text': tokenizer.apply_chat_template(thread, tokenize=False, chat_template=chat_template)})
            except Exception as e:
                print(f"ERROR: {e}")
                print(thread)
                import sys
                sys.exit(0)
            pbar.update(1)
    
    return threads