FROM registry.seculayer.com:31500/ape/python-base:py3.7 as builder
ARG app="/opt/app"

RUN sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
        gfortran openexr libatlas-base-dev \
        libtbb2 libtbb-dev libdc1394-22-dev
RUN mkdir -p $app
WORKDIR $app

COPY ./requirements.txt ./requirements.txt
RUN pip3.7 install --upgrade pip setuptools wheel
RUN pip3.7 install -r ./requirements.txt -t $app/lib

COPY ./xai ./xai
COPY ./setup.py ./setup.py

RUN pip3.7 install wheel
RUN python3.7 setup.py bdist_wheel

FROM registry.seculayer.com:31500/ape/python-base:py3.7 as app
ARG app="/opt/app"
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN mkdir -p /eyeCloudAI/app/ape/xai
WORKDIR /eyeCloudAI/app/ape/xai

COPY ./xai.sh /eyeCloudAI/app/ape/xai

COPY --from=builder "$app/lib" /eyeCloudAI/app/ape/xai/lib

COPY --from=builder "$app/dist/xai-1.0.0-py3-none-any.whl" \
        /eyeCloudAI/app/ape/xai/xai-1.0.0-py3-none-any.whl

RUN pip3.7 install /eyeCloudAI/app/ape/xai/xai-1.0.0-py3-none-any.whl --no-dependencies  \
    -t /eyeCloudAI/app/ape/xai \
    && rm /eyeCloudAI/app/ape/xai/xai-1.0.0-py3-none-any.whl

RUN groupadd -g 1000 aiuser
RUN useradd -r -u 1000 -g aiuser aiuser
RUN chown -R aiuser:aiuser /eyeCloudAI
USER aiuser

CMD []
