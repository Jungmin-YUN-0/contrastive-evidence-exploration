import re


def _whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def passage_key(title, text):
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return f"{_whitespace(str(title or ''))}\n{_whitespace(str(text or ''))}"


def build_doc_index_maps(docs):
    text_to_idx, title_to_idx = {}, {}
    for index, doc in enumerate(docs):
        title, _, body = doc.partition("\n")
        key = passage_key(title, body)
        text_to_idx.setdefault(key, []).append(index)
        title_to_idx.setdefault(key.split("\n", 1)[0], []).append(index)
    return text_to_idx, title_to_idx


def extract_gold_indices(sample, text_to_idx, title_to_idx):
    label = sample.get("id", sample.get("_id", sample.get("question", "?")))

    def fail(message):
        raise ValueError(f"Question {label!r}: {message}")

    gold = set()
    if "supporting_facts" in sample:
        titles = {_whitespace(str(fact[0])) for fact in sample["supporting_facts"]}
        represented = set()
        context_keys = {}
        for title, body in sample.get("context", []):
            normalized_title = _whitespace(str(title))
            if normalized_title in titles:
                key = passage_key(title, body)
                context_keys.setdefault(normalized_title, set()).add(key)
                gold.add(key)
                represented.add(normalized_title)
        missing = titles - represented
        if missing:
            fail(f"supporting titles absent from context: {sorted(missing)}")
        for title, keys in context_keys.items():
            if len(keys) > 1:
                fail(f"ambiguous supporting title {title!r}")
    else:
        field = "contexts" if "contexts" in sample else "paragraphs"
        for item in sample.get(field, []):
            supporting = item.get("is_supporting", field == "paragraphs")
            if not isinstance(supporting, bool):
                fail("is_supporting must be a JSON boolean")
            if supporting:
                gold.add(passage_key(item.get("title", ""),
                                     item.get("text", item.get("paragraph_text", ""))))
    if not gold:
        fail("no supporting passages were provided")

    indices = set()
    for key in sorted(gold):
        title, body = key.split("\n", 1)
        if not body:
            fail(f"supporting passage {title!r} has no body")
        matches = text_to_idx.get(key, [])
        if not matches:
            reason = ("title exists but its body does not match"
                      if title_to_idx.get(title) else "title is missing")
            fail(f"supporting passage {title!r} absent from corpus ({reason})")
        if len(matches) != 1:
            fail(f"ambiguous supporting passage {title!r}: matches corpus indices {matches}")
        indices.add(matches[0])
    return indices
