FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN     apt-get -qq update                      &&\
        apt-get -qq install build-essential     \
        python3-pip                             &&\
        pip3 install pycuda                     &&\
        pip3 install flask                      &&\
        pip3 install Pillow
COPY . /app
WORKDIR /app
EXPOSE 5000

CMD python3 ./convolucion.py