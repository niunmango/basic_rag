# RAG por línea de comandos

Pasos para usarlo:

## Lanzar el servidor de ollama

```bash
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm

docker exec -it ollama ollama pull llama3
```

## Instalar las depencias

```bash
pip install -r requirements.txt
```

## Ejecutar el script

```bash
python rag.py
```
Se termina la sesión con `/chau`

## Modificar el contenido y/o el modelo

+ El conocimiento que utiliza para contestar está en inscr.txt
+ Si existe el directorio embeddings, debe borrarse para usar otro modelo o cargar nuevo contenido
+ El modelo que ha dado mejores resultados es `llama3`. Para modificarlo es necesario hacer pull del nuevo modelo con `docker exec -it ollama ollama pull nuevo_modelo` y luego cambiar en `rag.py` el nombre del modelo en la variable `MODEL_NAME`.