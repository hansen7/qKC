# from https://github.com/rigetti/pyquil/blob/master/Dockerfile
# specify the dependency versions (can be overriden with --build_arg)
ARG quilc_version=1.20.0
ARG qvm_version=1.17.1
ARG python_version=3.6

# use multi-stage builds to independently pull dependency versions
FROM rigetti/quilc:$quilc_version as quilc
FROM rigetti/qvm:$qvm_version as qvm
FROM python:$python_version

WORKDIR /workspace/qkc
# copy over the pre-built quilc binary from the first build stage
COPY --from=quilc /src/quilc/quilc /src/quilc/quilc

# copy over the pre-built qvm binary from the second build stage
COPY --from=qvm /src/qvm/qvm /src/qvm/qvm
COPY ./docker/entrypoint.sh /src/

# install the missing apt packages that aren't copied over
RUN apt-get update && apt-get -yq dist-upgrade && \
    apt-get install --no-install-recommends -yq \
    git libblas-dev screen tmux libffi-dev \
    liblapack-dev libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*
    
# install ipython
RUN pip install --no-cache-dir ipython
RUN pip install pyyaml torch matplotlib h5py pyquil
# copy over files and install requirements
# ADD . /src/pyquil
# WORKDIR /src/pyquil
# RUN pip install -e .

# use an entrypoint script to add startup commands (qvm & quilc server spinup)
RUN chmod 777 /src/entrypoint.sh
ENTRYPOINT ["/src/entrypoint.sh"]
# CMD ["ipython"]
