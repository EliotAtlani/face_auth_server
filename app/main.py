from datetime import datetime, timedelta
import io
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

pc = Pinecone(api_key="2b7fe4b0-4b82-4d37-bab1-5f55d38f6758")
index = pc.Index("facial-recognition")


app = FastAPI()

# CORS
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

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


@app.post("/check-email/")
async def check_email(email: str = Form(...)):
    print("Email:", email)
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


@app.post("/validate-face/")
async def validate_face(file: UploadFile = File(...), email: str = Form(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Embbeding
    ibed = imgbeddings()

    embedding = ibed.to_embeddings(image)

    # Upsert embedding into Pinecone
    index.upsert(
        vectors=[{"id": "A", "values": embedding[0].tolist()}], namespace=email
    )

    return JSONResponse(
        content={"message": "Face validated successfully"}, status_code=200
    )


# Function to create JWT token
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

    # Embbedings
    ibed = imgbeddings()

    embedding = ibed.to_embeddings(cropped_face)

    result = index.query(
        namespace=email,
        vector=embedding[0].tolist(),
        top_k=1,
        include_values=False,
    )

    print(result)
    if len(result["matches"]) == 0:
        print("Face not authenticated")
        return JSONResponse(
            content={"message": "Face not authenticated"}, status_code=404
        )

    if result["matches"][0]["score"] < 0.92:
        print("Face not authenticated")
        return JSONResponse(
            content={"message": "Face not authenticated"}, status_code=401
        )

    # Create JWT token
    access_token_expires = timedelta(minutes=float(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )

    print("Face authenticated successfully", access_token)

    return JSONResponse(
        content={
            "message": "Face authenticated successfully",
            "access_token": access_token,
        },
        status_code=200,
    )
