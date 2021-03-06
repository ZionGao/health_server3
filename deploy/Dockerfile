FROM ubuntu:latest
ENV SERVICE_HOME=/home/health_server
ENV SOURCE_PATH=${SERVICE_HOME}/source
ENV MODEL_PATH=${SERVICE_HOME}/model
ENV COMMON_PATH=${SERVICE_HOME}/common
ENV DATA_PATH=${SERVICE_HOME}/data
ENV PATH=$PATH:$SERVICE_HOME
ENV LANG=C.UTF-8
LABEL maintainer="Zion.Gao@foxmail.com"

COPY deploy/sources.list /etc/apt/
RUN apt-get update && \
    apt-get install build-essential -y && \
    apt-get install gcc libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y

COPY deploy/anaconda.sh /
RUN /bin/bash /anaconda.sh -b -p /opt/conda

RUN mkdir ~/.pip
COPY deploy/pip.conf /
RUN cp /pip.conf ~/.pip

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    ln -s /opt/conda/bin/python3.7 /usr/bin/python && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN buildDeps='gcc'

RUN /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Flask && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask_cors && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple easydict && \
    /opt/conda/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn && \
    mkdir -p $SERVICE_HOME && \
    mkdir -p $SOURCE_PATH && \
    mkdir -p $MODEL_PATH && \
    mkdir -p $COMMON_PATH && \
    mkdir -p $DATA_PATH

COPY model $MODEL_PATH
COPY data $DATA_PATH
COPY common $COMMON_PATH
COPY source $SOURCE_PATH

COPY docker-entrypoint.sh $SERVICE_HOME

WORKDIR $SOURCE_PATH
EXPOSE 9003
ENTRYPOINT ["bash", "docker-entrypoint.sh"]

