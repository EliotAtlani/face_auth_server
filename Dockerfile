# 
FROM python:3.10-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir -r requirements.txt

# 
COPY ./app /code/app

# 
CMD ["fastapi", "run", "app/main.py"]