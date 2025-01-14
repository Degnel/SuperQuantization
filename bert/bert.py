import json
import os
from bert.__main__ import train
from bert.dataset.vocab import build
import argparse


def bert():
    json_path = "./data/squad/train-v2.0.json"
    txt_path = "./data/squad/squad.txt"

    if not os.path.exists(txt_path):
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            for article in data["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        answers = [answer["text"] for answer in qa["answers"]]
                        for answer in answers:
                            if answer:
                                txt_file.write(f"{context} {question}\t{answer}\n")

    with open(txt_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    valid_lines = [line for line in lines if line.count("\t") == 1]

    with open(txt_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(valid_lines)
