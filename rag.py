import numpy as np
import ollama
import os
import json
from numpy.linalg import norm
import time

MODEL_NAME = "llama3"

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings

def find_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = f"""Eres un chatbot de atención al alumno que contesta basándose en el contexto.
                       Solo contestar con lo que existe en el contexto.
                       Si no estás seguro di que no lo sabes.
                       Respuestas cortas en el idioma en el que te preguntan.
                       Ahorrar palabras.
                       Contexto:
                       """

    filename = "inscr.txt"
    paragraphs = parse_file(filename)
    
    embeddings = get_embeddings(filename, MODEL_NAME, paragraphs)

    while True:
        prompt = input("¿Cómo puedo ayudarte? ")
        if prompt.lower() == "/chau":
            break

        # Start timer
        start_time = time.time()

        prompt_embedding = ollama.embeddings(model=MODEL_NAME, prompt=prompt)["embedding"]
        # most similar results
        most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n".join(paragraphs[item[1]] for item in most_similar_chunks)
                },
                {
                    "role": "user", "content": prompt
                }
            ]
        )

        # End timer and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(response["message"]["content"])
        print(f"Respuesta obtenida en {elapsed_time:.2f} segundos.")  # Print time with 2 decimal places


if __name__ == "__main__":
    main()
