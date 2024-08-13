ARG BASE_IMAGE="registry.renkulab.io/kirschnj/docker-images/main/runai-py:2024-05-03"

FROM ${BASE_IMAGE}

USER root

# add python dependencies
COPY docker/runai/environment.yml /tmp/environment.yml
COPY docker/runai/requirements.txt /tmp/requirements.txt

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate default && \
    mamba env update -f /tmp/environment.yml -n default && \
    mamba clean -y --all && \
    pip install -r /tmp/requirements.txt --no-cache-dir &&\
    rm /tmp/environment.yml /tmp/requirements.txt

USER ${NB_USER}
