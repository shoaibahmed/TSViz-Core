FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM alpine:latest

MAINTAINER Shoaib Ahmed Siddiqui <shoaib_ahmed.siddiqui@dfki.de>

RUN apt update && apt install -y \
    build-essential \
    curl \
    git \
    wget \
    libjpeg-dev \
    openjdk-8-jdk \
    virtualenv \
    && rm -rf /var/lib/lists/*

# Install Anaconda
RUN wget "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "miniconda.sh" && \
    bash "miniconda.sh" -b -p "/conda" && \
    rm miniconda.sh && \
    echo PATH='/conda/bin:$PATH' >> /root/.bashrc && \
    /conda/bin/conda config --add channels conda-forge && \
    /conda/bin/conda update --yes -n base conda && \
    /conda/bin/conda update --all --yes

#RUN /conda/bin/conda create -n py36 python=3.6 anaconda
#RUN ln -s /conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
#RUN echo "conda activate" >> ~/.bashrc
#RUN /conda/bin/conda activate py36
RUN /conda/bin/conda install python=3.6

# Install TensorFlow
RUN /conda/bin/pip install \
    tensorflow-gpu==1.13.1 \
    keras==2.2.4 \
    flask \
    flask-cors \
    scikit-learn \
    pandas \
    tqdm \
    matplotlib \
    seaborn \
    scipy \
    numpy \
    kerassurgeon \
    cython

# Install DTAI Distance from sources
RUN git clone https://github.com/wannesm/dtaidistance
WORKDIR /dtaidistance
RUN /conda/bin/python setup.py build_ext --inplace
RUN /conda/bin/python setup.py install

# Copy the TSViz resources
WORKDIR /root/TSViz/

COPY  ./src/visualizationService.py ./
COPY  ./src/tsviz.py ./
COPY  ./src/utils.py ./
COPY  ./src/config.py ./

# Copy the datasets
COPY  ./src/anomaly_dataset.pickle ./
COPY  ./src/datamark-internettrafficdata.csv ./

# Copy the CNN models
COPY  ./src/cnn_anomaly_dataset.h5 ./
COPY  ./src/modCNN_Outsidesteps1e5000lr0.001lb50batch5datamark-internettrafficdata.ctrs14000tes4000deriv1.h5 ./

COPY ./src/start.sh ./
RUN chmod +x ./start.sh

# Automatically execute the service upon the start of the container
ENTRYPOINT ["/root/TSViz/start.sh"]
