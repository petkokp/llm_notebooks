def doc_to_text(doc):
    question = (doc["question"] or "").strip()
    choices = (doc["choices"] or [''] * 4)[:4] + [''] * (4 - len(doc["choices"] or []))

    return (
        f"{question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Отговор:"
    )
