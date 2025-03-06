FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update
RUN apt-get -y install tmux
RUN apt-get -y install nano

RUN python -m pip install --upgrade pip
RUN python -m pip install "jax[cuda12]" rich flax tqdm

#COPY ./requirements.txt .
#RUN pip install -r ./requirements.txt