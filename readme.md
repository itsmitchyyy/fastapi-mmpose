### Prerequisites

- Python
- CUDA 11.7

### Installation

```
docker build -f Dockerfile -t fastapi-mmpose .
```

### Running the server

```
docker run -p 3000:3000 --gpus all -it fastapi-mmpose
```
