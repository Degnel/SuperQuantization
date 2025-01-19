from datasets import load_dataset
from collections import Counter
import torch
import os
import pickle as pkl


def get_data(seq_length=5, vocab_size=10000, force_create=False):
    data_dir = "./data/tiny_stories"
    data_files = {
        "X_train": os.path.join(data_dir, "X_train.pt"),
        "y_train": os.path.join(data_dir, "y_train.pt"),
        "X_test": os.path.join(data_dir, "X_test.pt"),
        "y_test": os.path.join(data_dir, "y_test.pt"),
        "vocab": os.path.join(data_dir, "vocab.pkl"),
    }

    # Vérifier si les fichiers existent
    if all(os.path.exists(file) for file in data_files.values()) and not force_create:
        # Charger les données depuis les fichiers
        X_train = torch.load(data_files["X_train"])
        y_train = torch.load(data_files["y_train"])
        X_test = torch.load(data_files["X_test"])
        y_test = torch.load(data_files["y_test"])
        with open(data_files["vocab"], "rb") as f:
            vocab = pkl.load(f)
    else:
        # Chargement du dataset TinyStories
        ds = load_dataset("roneneldan/TinyStories")

        # Extraction des textes d'entraînement et de validation
        train_texts = ds["train"]["text"]
        validation_texts = ds["validation"]["text"]

        # Prétraitement des données
        # Étape 1 : Découpage en mots
        train_tokens = [text.split() for text in train_texts]
        validation_tokens = [text.split() for text in validation_texts]

        # Étape 2 : Comptage des mots les plus fréquents
        all_train_tokens = [word for tokens in train_tokens for word in tokens]
        word_counter = Counter(all_train_tokens)
        most_common_words = [word for word, _ in word_counter.most_common(vocab_size)]

        # Création d'un vocabulaire où chaque mot a un numéro de token
        vocab = {word: idx for idx, word in enumerate(most_common_words)}

        # Conversion des textes en séquences de tokens
        tokenized_train = [tokenize(tokens, vocab) for tokens in train_tokens]
        tokenized_validation = [tokenize(tokens, vocab) for tokens in validation_tokens]

        X_train, y_train = create_sequences(tokenized_train, seq_length)
        X_test, y_test = create_sequences(tokenized_validation, seq_length)

        # Enregistrer les données dans des fichiers
        os.makedirs(data_dir, exist_ok=True)
        torch.save(X_train, data_files["X_train"])
        torch.save(y_train, data_files["y_train"])
        torch.save(X_test, data_files["X_test"])
        torch.save(y_test, data_files["y_test"])
        with open(data_files["vocab"], "wb") as f:
            pkl.dump(vocab, f)

    return X_train, y_train, X_test, y_test, vocab


# Fonction pour convertir les mots en tokens
def tokenize(tokens, vocab, unk_token=0):
    return [vocab.get(word, unk_token) for word in tokens]


# Création des séquences X (entrée) et y (cible)
def create_sequences(tokenized_texts, sequence_length):
    X, y = [], []
    for tokens in tokenized_texts:
        if len(tokens) > sequence_length:  # Ignore les textes trop courts
            for i in range(len(tokens) - sequence_length):
                X.append(tokens[i : i + sequence_length])
                y.append(tokens[i + 1 : i + sequence_length + 1])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
