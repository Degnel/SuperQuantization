from datasets import load_dataset
from collections import Counter
import os
import pickle as pkl
from tests.text_generation.tiny_stories_dataset import TinyStoriesDataset


def get_data(seq_length=5, vocab_size=10000, force_create=False):
    data_dir = "./data/tiny_stories"
    data_files = {
        "tokenized_train": os.path.join(data_dir, "tokenized_train.pkl"),
        "tokenized_validation": os.path.join(data_dir, "tokenized_validation.pkl"),
        "vocab": os.path.join(data_dir, "vocab.pkl"),
    }

    # Vérifier si les fichiers existent
    if all(os.path.exists(file) for file in data_files.values()) and not force_create:
        # Charger les données depuis les fichiers
        with open(data_files["tokenized_train"], "rb") as f:
            tokenized_train = pkl.load(f)
        with open(data_files["tokenized_validation"], "rb") as f:
            tokenized_validation = pkl.load(f)
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
        vocab = {word: idx for idx, word in enumerate(most_common_words, 1)}

        # Conversion des textes en séquences de tokens
        tokenized_train = [tokenize(tokens, vocab) for tokens in train_tokens]
        tokenized_validation = [tokenize(tokens, vocab) for tokens in validation_tokens]

        # X_train, y_train = create_sequences(tokenized_train, seq_length)
        # X_test, y_test = create_sequences(tokenized_validation, seq_length)

        # Enregistrer les données dans des fichiers
        os.makedirs(data_dir, exist_ok=True)
        with open(data_files["tokenized_train"], "wb") as f:
            pkl.dump(tokenized_train, f)
        with open(data_files["tokenized_validation"], "wb") as f:
            pkl.dump(tokenized_validation, f)
        with open(data_files["vocab"], "wb") as f:
            pkl.dump(vocab, f)

    train_dataset = TinyStoriesDataset(tokenized_train, seq_length)
    validation_dataset = TinyStoriesDataset(tokenized_train, seq_length)
        
    return train_dataset, validation_dataset, vocab


# Fonction pour convertir les mots en tokens
def tokenize(tokens, vocab, unk_token=0):
    return [vocab.get(word, unk_token) for word in tokens]
