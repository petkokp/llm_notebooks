from typing import Iterable

def iter_texts(ds) -> Iterable[str]:
    for ex in ds:
        if "text" in ex:
            yield ex["text"] # for 'chitanka' dataset
        if "messages" in ex:
            for t in ex["messages"]:
                for c in t.get("content", []):
                    if c.get("type") == "text" and c.get("text"):
                        yield c["text"]
        elif "texts" in ex:
            for qa in ex["texts"]:
                if qa.get("user"):      yield qa["user"]
                if qa.get("assistant"): yield qa["assistant"]