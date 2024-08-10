<p align="center">
  <a href="https://faceauth.eliotatlani.fr/" target="blank"><img src="https://faceauth.eliotatlani.fr/assets/logo-DFPyzX_F.png" width="200" alt="Faceauth Logo" /></a>
</p>

## Description - FaceAuth Server

FaceAuth is a facial recognition system that allows you to connect to your accounts without having to enter a password.

The FaceAuth server is built with [FastAPI](https://fastapi.tiangolo.com) using Websocket.

Images are send to the server and converted to embeddings. Thoses embeddings are store in a [PineconeAI](https://login.pinecone.io) vector store database. When a user try to log in, we transform this face into an embeddings and run a similarity check with the others vectors in the database.
 de
NB: No images are stored or saved

## Installation

Create an isolated python environment 
```bash
python3 -m venv myenv
```

On Windows

```bash
myenv\Scripts\activate
```

On macOS/Linux 

```bash
source myenv/bin/activate
```

Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Running the app

```bash
python3 app/main.py 
```

## With Docker

Build the image

```bash
docker build -t <image-name> .
```

Run the container
```bash
docker run -d -p 8000:8000 <image-name>
```
