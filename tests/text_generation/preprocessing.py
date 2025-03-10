from datasets import load_dataset
from collections import Counter
import os
import pickle as pkl
from tests.text_generation.text_dataset import TextDataset
import re
from tqdm import tqdm


def split_camel_case(token):
    return re.findall(r"[a-z]+|[A-Z][a-z]*", token)


def split(text):
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)

    final_tokens = []
    for token in tokens:
        if token.isidentifier():
            split_tokens = split_camel_case(token)
            if "_" in token:
                new_token = []
                for part in re.split(r"(_+)", token):
                    if part:
                        new_token.append(part)
                final_tokens.extend(new_token)
            else:
                final_tokens.extend(split_tokens)
        else:
            final_tokens.append(token)

    return final_tokens


def tokenize(tokens, vocab, unk_token=0):
    return [vocab.get(word, unk_token) for word in tokens]


def get_data(
    seq_length=5,
    vocab_size=10000,
    train_max_batch_count=1000,
    test_max_batch_count=100,
    step=1,
    force_create=False,
    wiki=False
):
    data_dir = "./data/wikipedia" if wiki else "./data/codeparrot"
    data_files = {
        "tokenized_train": os.path.join(data_dir, "tokenized_train.pkl"),
        "tokenized_validation": os.path.join(data_dir, "tokenized_validation.pkl"),
        "vocab": os.path.join(data_dir, "vocab.pkl"),
    }

    # Vérifier si les fichiers existent
    if all(os.path.exists(file) for file in data_files.values()) and not force_create:
        print("Fetching data...")
        # Charger les données depuis les fichiers
        with open(data_files["tokenized_train"], "rb") as f:
            tokenized_train = pkl.load(f)
        with open(data_files["tokenized_validation"], "rb") as f:
            tokenized_validation = pkl.load(f)
        with open(data_files["vocab"], "rb") as f:
            vocab = pkl.load(f)
    else:
        print("Loading datasets...")
        train_texts = load_dataset(
            "codeparrot/codeparrot-train-v2-near-dedup", split="train[:1000]"
        )["content"]
        validation_texts = load_dataset(
            "codeparrot/codeparrot-valid-v2-near-dedup", split="train[:1000]"
        )["content"]

        print("Splitting training data...")
        train_tokens = [
            split(text) for text in tqdm(train_texts, desc="Tokenizing train data")
        ]

        print("Splitting validation data...")
        validation_tokens = [
            split(text)
            for text in tqdm(validation_texts, desc="Tokenizing validation data")
        ]

        print("Building vocabulary...")
        all_train_tokens = [
            word
            for tokens in tqdm(train_tokens, desc="Flattening tokens")
            for word in tokens
        ]
        word_counter = Counter(all_train_tokens)
        most_common_words = [word for word, _ in word_counter.most_common(vocab_size)]

        vocab = {word: idx for idx, word in enumerate(most_common_words, 1)}

        print("Tokenizing training data...")
        tokenized_train = [
            tokenize(tokens, vocab)
            for tokens in tqdm(train_tokens, desc="Tokenizing train")
        ]

        print("Tokenizing validation data...")
        tokenized_validation = [
            tokenize(tokens, vocab)
            for tokens in tqdm(validation_tokens, desc="Tokenizing validation")
        ]

        os.makedirs(data_dir, exist_ok=True)
        with open(data_files["tokenized_train"], "wb") as f:
            pkl.dump(tokenized_train, f)
        with open(data_files["tokenized_validation"], "wb") as f:
            pkl.dump(tokenized_validation, f)
        with open(data_files["vocab"], "wb") as f:
            pkl.dump(vocab, f)

    train_dataset = TextDataset(
        tokenized_train, seq_length, step, train_max_batch_count
    )
    validation_dataset = TextDataset(
        tokenized_validation, seq_length, step, test_max_batch_count
    )

    return train_dataset, validation_dataset, vocab
