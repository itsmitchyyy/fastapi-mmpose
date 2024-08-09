### Prerequisites

- Python
- CUDA 11.7

### Installation

Create a folder named `output` in the project root directory.

then build the docker image

```
docker build -f Dockerfile -t fastapi-mmpose .
```

### Running the server

```
docker run -p 3000:3000 --gpus all -it fastapi-mmpose
```
