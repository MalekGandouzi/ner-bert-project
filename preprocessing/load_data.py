def load_conll(path):
    sentences = []
    labels = []

    words = []
    tags = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
            else:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)

        # dernière phrase
        if words:
            sentences.append(words)
            labels.append(tags)

    return sentences, labels
