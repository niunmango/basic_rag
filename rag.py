import numpy as np
import ollama
import os
import json
from numpy.linalg import norm

MODEL_NAME = "gemma:2b"

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
    # create dir if it doesnt exist
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
    with open (f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)['embedding']
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
    SYSTEM_PROMPT = """Eres un secretario de atencion al alumno que contesta preguntas
                        en un chat online basandose en el contexto creado.
                        Puedes saludar. Luego solo contestar con lo que existe en el contexto.
                        Si no estás seguro, di que no lo sabes.
                        Respuestas concretas y cortas siempre en el idioma en el que te preguntan.
                        Contexto:
                        """

    filename = "inscr.txt"
    paragraphs = parse_file(filename)
    
    embeddings = get_embeddings(filename, MODEL_NAME, paragraphs)

    prompt = input("¿Cómo puedo ayudarte? ")
    prompt_embedding = ollama.embeddings(model=MODEL_NAME, prompt=prompt)[
        "embedding"
        ]
    # most similar results
    most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT +"\n".join(paragraphs[item[1]] for item in most_similar_chunks)
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
    
    print(response["message"]["content"])
        

if __name__ == "__main__":
    main()