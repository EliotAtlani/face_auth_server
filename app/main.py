from datetime import datetime, timedelta
import io
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from imgbeddings import imgbeddings
from PIL import Image
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from .helpers.crop_image import crop_image
from .helpers.detect_faces import detect_faces
import os
import jwt
import uuid
from fastapi.middleware.cors import CORSMiddleware
import json
import base64

########################
### Load Environment ###
########################

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("facial-recognition")


################
### FastAPI ###
################

app = FastAPI()

#######################
### CORS Middleware ###
#######################

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


#################
### Endpoints ###
#################


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/ping/")
async def ping():
    return {"message": "pong"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await process_websocket_data(websocket, data)
    except WebSocketDisconnect:
        print("WebSocket connection closed")


async def process_websocket_data(websocket: WebSocket, data: str):
    try:
        json_data = json.loads(data)
        email = json_data.get("email")
        image_data = json_data.get("imageData")

        if not email or not image_data:
            await websocket.send_text(
                json.dumps({"success": False, "error": "Missing email or image data"})
            )
            return

        # Decode base64 image
        image_data = image_data.split(",")[
            1
        ]  # Remove the "data:image/jpeg;base64," part
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        faces, img = detect_faces(image)

        if len(faces) == 0:
            await websocket.send_text(
                json.dumps({"success": False, "error": "No faces found"})
            )
            return

        cropped_face = crop_image(img, faces[0])

        # Embedding
        ibed = imgbeddings()
        embedding = ibed.to_embeddings(cropped_face)

        result = index.query(
            namespace=email,
            vector=embedding[0].tolist(),
            top_k=5,
            include_values=False,
        )

        if len(result["matches"]) == 0:
            await websocket.send_text(
                json.dumps(
                    {"success": False, "error": "Face not authenticated", "stop": True}
                )
            )
            return

        if len(result["matches"]) < 5:
            mean = np.mean(
                [result["matches"][i]["score"] for i in range(len(result["matches"]))]
            )
        else:
            mean = np.mean([result["matches"][i]["score"] for i in range(5)])

        print("MEAN", mean)
        if mean < 0.92:
            await websocket.send_text(json.dumps({"error": "Face not authenticated"}))
            return

        # Create JWT token
        access_token_expires = timedelta(minutes=float(ACCESS_TOKEN_EXPIRE_MINUTES))
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )

        add_new_images_to_index(email, embedding[0].tolist())

        await websocket.send_text(
            json.dumps(
                {
                    "success": True,
                    "message": "Face authenticated successfully",
                    "access_token": access_token,
                }
            )
        )

    except Exception as e:
        print(f"Error processing WebSocket data: {str(e)}")
        await websocket.send_text(
            json.dumps(
                {
                    "success": False,
                    "message": f"Error processing WebSocket data: {str(e)}",
                }
            )
        )


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


def add_new_images_to_index(email: str, embedding):
    uniqueId = str(uuid.uuid4())
    # Upsert all embeddings into Pinecone
    index.upsert(vectors=[{"id": uniqueId, "values": embedding}], namespace=email)
    print("Added new image to index")


# @app.post("/auth-face/")
# async def auth_face(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     email: str = Form(...),
# ):
#     # Read the image file
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))

#     faces, img = detect_faces(image)

#     if len(faces) == 0:
#         print("No faces found")
#         raise HTTPException(status_code=404, detail="No faces found")

#     cropped_face = crop_image(img, faces[0])

#     # Embbeding
#     ibed = imgbeddings()

#     embedding = ibed.to_embeddings(cropped_face)

#     result = index.query(
#         namespace=email,
#         vector=embedding[0].tolist(),
#         top_k=5,
#         include_values=False,
#     )

#     print(result)

#     if len(result["matches"]) == 0:
#         print("Face not authenticated")
#         return JSONResponse(
#             content={"message": "Face not authenticated"}, status_code=403
#         )

#     if len(result["matches"]) < 5:
#         mean = np.mean(
#             [result["matches"][i]["score"] for i in range(len(result["matches"]))]
#         )
#     else:
#         mean = np.mean([result["matches"][i]["score"] for i in range(5)])

#     print("MEAN", mean)
#     if mean < 0.92:
#         print("Face not authenticated")
#         return JSONResponse(
#             content={"message": "Face not authenticated"}, status_code=401
#         )

#     # Create JWT token
#     access_token_expires = timedelta(minutes=float(ACCESS_TOKEN_EXPIRE_MINUTES))
#     access_token = create_access_token(
#         data={"sub": email}, expires_delta=access_token_expires
#     )

#     background_tasks.add_task(add_new_images_to_index, email, embedding[0].tolist())

#     return JSONResponse(
#         content={
#             "message": "Face authenticated successfully",
#             "access_token": access_token,
#         },
#         status_code=200,
#     )
