from datetime import datetime, timedelta
import io
from typing import List
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from imgbeddings import imgbeddings
from PIL import Image
from pinecone.grpc import PineconeGRPC as Pinecone
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from .helpers.crop_image import crop_image
from .helpers.detect_faces import detect_faces
import os
import jwt

# Load environment variables from .env file
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("facial-recognition")


app = FastAPI()

# CORS
from fastapi.middleware.cors import CORSMiddleware

environment = os.getenv("ENVIRONMENT")

if environment == "development":
    origins = ["*"]  # Allow all origins in development
else:
    origins = ["https://www.eliotatlani.fr"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/ping/")
async def ping():
    return {"message": "pong"}


@app.post("/check-email/")
async def check_email(email: str = Form(...)):
    result = index.query(
        namespace=email,
        vector=np.random.rand(768).tolist(),
        top_k=1,
        include_values=False,
    )

    if len(result["matches"]) > 0:
        return JSONResponse(content={"message": "Email already exist"}, status_code=404)
    return JSONResponse(content={"message": "Email available"}, status_code=200)


@app.post("/detect-faces/")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    faces, img = detect_faces(image)

    if len(faces) == 0:
        print("No faces found")
        return JSONResponse(content={"message": "No faces found"}, status_code=404)

    cropped_face = crop_image(img, faces[0])

    # Convert cropped face to bytes
    buf = io.BytesIO()
    cropped_face.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/validate-face-batch/")
async def validate_face_batch(
    file_0: UploadFile = File(None),
    file_1: UploadFile = File(None),
    file_2: UploadFile = File(None),
    file_3: UploadFile = File(None),
    file_4: UploadFile = File(None),
    file_5: UploadFile = File(None),
    file_6: UploadFile = File(None),
    file_7: UploadFile = File(None),
    file_8: UploadFile = File(None),
    file_9: UploadFile = File(None),
    email: str = Form(...),
):
    ibed = imgbeddings()
    embeddings = []

    files = [
        file_0,
        file_1,
        file_2,
        file_3,
        file_4,
        file_5,
        file_6,
        file_7,
        file_8,
        file_9,
    ]

    for i, file in enumerate(files):
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            faces, img = detect_faces(image)

            if len(faces) == 0:
                return
            cropped_face = crop_image(img, faces[0])

            # Generate embedding for each image
            embedding = ibed.to_embeddings(cropped_face)
            embeddings.append(embedding[0].tolist())

    if not embeddings:
        return JSONResponse(
            content={"message": "No valid images were provided"}, status_code=400
        )

    # Upsert all embeddings into Pinecone
    vectors = [{"id": f"image_{i}", "values": emb} for i, emb in enumerate(embeddings)]
    index.upsert(vectors=vectors, namespace=email)

    return JSONResponse(
        content={
            "message": f"Processed {len(embeddings)} images and stored embeddings successfully"
        },
        status_code=200,
    )


# Function to create JWT tokens
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=float(ACCESS_TOKEN_EXPIRE_MINUTES)
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/auth-face/")
async def auth_face(file: UploadFile = File(...), email: str = Form(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    faces, img = detect_faces(image)

    if len(faces) == 0:
        print("No faces found")
        raise HTTPException(status_code=404, detail="No faces found")

    cropped_face = crop_image(img, faces[0])

    # Embbeding
    ibed = imgbeddings()

    embedding = ibed.to_embeddings(cropped_face)

    result = index.query(
        namespace=email,
        vector=embedding[0].tolist(),
        top_k=5,
        include_values=False,
    )

    if len(result["matches"]) == 0:
        print("Face not authenticated")
        return JSONResponse(
            content={"message": "Face not authenticated"}, status_code=404
        )

    # Compute the mean of the best 5 result
    mean = np.mean([result["matches"][i]["score"] for i in range(5)])

    print("MEAN", mean)
    if mean < 0.95:
        print("Face not authenticated")
        return JSONResponse(
            content={"message": "Face not authenticated"}, status_code=401
        )

    # Create JWT token
    access_token_expires = timedelta(minutes=float(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )

    return JSONResponse(
        content={
            "message": "Face authenticated successfully",
            "access_token": access_token,
        },
        status_code=200,
    )
