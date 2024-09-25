def get_vocabulary(text: str):
    return sorted(set(text))

def get_encoding(vocabulary):
    return { c: i for i, c in enumerate(vocabulary) }
    
def get_decoding(vocabulary):
    return { i: c for i, c in enumerate(vocabulary) }

def encode(text: str, encoding):
    return [encoding[c] for c in text]

test_text = "франкейщайн!"
test_encoding = encode(test_text)

def decode(arr, decoding):
    return "".join([decoding[t] for t in arr])
