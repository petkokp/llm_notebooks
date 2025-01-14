TRANSLATION_REPLACE_PHRASE = "(това е само пример)"

def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]

def doc_to_target(doc):
    # check if "_" exists in the sentence
    if "_" in doc["sentence"]:
        idx = doc["sentence"].index("_") + 1
        return doc["sentence"][idx:].strip()
    elif "(това е само пример)" in doc["sentence"]:
        # find the phrase added during the winogrande dataset translation and return the text after it
        idx = doc["sentence"].index(TRANSLATION_REPLACE_PHRASE) + len(TRANSLATION_REPLACE_PHRASE)
        return doc["sentence"][idx:].strip()
    else:
        # if neither "_" nor "(това е само пример)" is found, return the whole sentence
        return doc["sentence"].strip()

def doc_to_choice(doc):
    # check if "_" exists in the sentence
    if "_" in doc["sentence"]:
        idx = doc["sentence"].index("_")
        options = [doc["option1"], doc["option2"]]
        return [doc["sentence"][:idx] + opt + doc["sentence"][idx+1:] for opt in options]
    elif TRANSLATION_REPLACE_PHRASE in doc["sentence"]:
        # replace the specific phrase added during the winogrande dataset translation
        idx = doc["sentence"].index(TRANSLATION_REPLACE_PHRASE)
        options = [doc["option1"], doc["option2"]]
        return [doc["sentence"][:idx] + opt + doc["sentence"][idx+len(TRANSLATION_REPLACE_PHRASE):] for opt in options]
    else:
        # If neither "_" nor "(това е само пример)" is found, return the original sentence unchanged
        return [doc["sentence"]]
