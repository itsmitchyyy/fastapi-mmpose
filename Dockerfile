ARG BASE_IMAGE=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}


ARG PROJECT_NAME=mmpose-fastapi
ARG USER_NAME=user
ARG GROUP_NAME=user
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.10
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"
ARG PORT=3000
ENV PORT_NUMBER=${PORT}

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

RUN apt update && apt install --no-install-recommends -y software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa

# Install necessary packages
RUN apt update && apt install --no-install-recommends -y \
    git sudo wget curl make cmake ssh openssh-client ffmpeg libgl1-mesa-dev\
    libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    build-essential libssl-dev libffi-dev \
    python${PYTHON_VERSION} python3-pip python-is-python3 python${PYTHON_VERSION}-distutils python${PYTHON_VERSION}-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

RUN python3 -m pip install --upgrade pip setuptools requests\ 
    && python3 -m pip install poetry cython xtcocotools openmim torch torchvision pillow jupyterlab pytest fastapi uvicorn python-multipart aiofiles numpy-quaternion

# RUN python3 -m pip install --upgrade pip setuptools requests
# RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN mim install mmengine "mmcv==2.1.0"
RUN mim install "mmpose==1.3.1"
RUN mim install "mmdet==3.2.0"

RUN groupadd --gid $GID $GROUP_NAME && \
    useradd --uid $UID --gid $GID -m $USER_NAME && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

WORKDIR ${APPLICATION_DIRECTORY}

COPY . .
RUN chown -R ${UID}:${GID} ${APPLICATION_DIRECTORY}

USER ${USER_NAME}

EXPOSE ${PORT_NUMBER}

CMD uvicorn main:app --host 0.0.0.0 --port $PORT_NUMBER

