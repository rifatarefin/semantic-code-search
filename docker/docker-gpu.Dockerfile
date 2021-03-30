ARG VER=latest
ARG ORG=tacc
FROM ${ORG}/tacc-ubuntu18:${VER}

########################################
# BUILD Args
########################################
ARG FLAGS
ARG VER
ARG REL
########################################
# Configure ENV
########################################
ENV CFLAGS=${FLAGS}
ENV CXXFLAGS=${FLAGS}
RUN RF=/etc/${REL}-release; echo ${VER} > $RF && chmod a+r $RF

########################################
# Install mpi
########################################

# necessities and IB stack
RUN apt-get update && apt-get install -yq gnupg2 ca-certificates
RUN curl -k -L http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN curl -k -L https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu18.04/mellanox_mlnx_ofed.list > /etc/apt/sources.list.d/mlnx_ofed.list
RUN apt-get update && \
    apt-get install -yq --no-install-recommends gfortran bison libibverbs-dev libnuma-dev \
	libibmad-dev libibumad-dev librdmacm-dev libxml2-dev ca-certificates libfabric-dev \
        mlnx-ofed-hpc ucx \
	&& docker-clean

# Install PSM2
ARG PSM=PSM2
ARG PSMV=11.2.78
ARG PSMD=opa-psm2-${PSM}_${PSMV}

RUN curl -L https://github.com/intel/opa-psm2/archive/${PSM}_${PSMV}.tar.gz | tar -xzf - \
    && cd ${PSMD} \
    && make PSM_AVX=1 -j $(nproc --all 2>/dev/null || echo 2) \
    && make LIBDIR=/usr/lib/x86_64-linux-gnu install \
    && cd ../ && rm -rf ${PSMD}

# Install impi-19.0.7
ARG MAJV=19
ARG MINV=0
ARG BV=.7
ARG DIR=intel${MAJV}-${MAJV}.${MINV}${BV}

RUN curl -k -L https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add -
RUN echo deb https://apt.repos.intel.com/mpi all main > /etc/apt/sources.list.d/intel-mpi.list
RUN apt-get update \
    && apt-get install -y intel-mpi-20${MAJV}${BV}-102 \
    && docker-clean

# Configure environment for impi
ENV MPIVARS_SCRIPT=/opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh \
    I_MPI_LIBRARY_KIND=release_mt \
    I_MPI_OFI_LIBRARY_INTERNAL=1 \
    I_MPI_REMOVED_VAR_WARNING=0 \
    I_MPI_VAR_CHECK_SPELLING=0 \
    BASH_ENV=/opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh
RUN sed -i 's~bin/sh~bin/bash~' $MPIVARS_SCRIPT \
    && sed -i '/bin\/bash/a \[ "${IMPI_LOADED}" == "1" \] && return' $MPIVARS_SCRIPT \
    && echo "export IMPI_LOADED=1" >> $MPIVARS_SCRIPT \
    && echo -e '#!/bin/bash\n. /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh -ofi_internal=1 release_mt\nexec "${@}"' > /entry.sh \
    && chmod +x /entry.sh

# Add hello world
ADD extras/hello.c /tmp/hello.c
RUN mpicc /tmp/hello.c -o /usr/local/bin/hellow \
    && rm /tmp/hello.c \
    && docker-clean

# Build benchmark programs
ADD extras/install_benchmarks.sh /tmp/install_benchmarks.sh
RUN bash /tmp/install_benchmarks.sh

# Test installation
RUN mpirun -n 2 hellow

FROM tensorflow/tensorflow:2.2.2-gpu-py3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN add-apt-repository -y ppa:git-core/ppa
RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    byobu \
    ca-certificates \
    git-core git \
    htop \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcupti-dev \
    openssh-server \
    python3.6 \
    python3.6-dev \
    software-properties-common \
    vim \
    unzip \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN apt-get -y update

#  Setup Python 3.6 (Need for other dependencies)
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install -y python3-setuptools
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip install --upgrade pip

# Pin TF Version on v1.12.0
RUN pip --no-cache-dir install tensorflow
# https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl

# Other python packages
RUN pip install --upgrade \
    altair==3.2.0 \
    annoy==1.16.0 \
    docopt==0.6.2 \
    dpu_utils==0.2.17 \
    ipdb==0.12.2 \
    jsonpath_rw_ext==1.2.2 \
    jupyter==1.0.0 \
    more_itertools==7.2.0 \
    numpy==1.16.5 \
    pandas==0.25.0 \
    parso==0.5.1 \
    pygments==2.4.2 \
    pyyaml==5.3 \
    requests==2.22.0 \
    scipy==1.3.1 \
    SetSimilaritySearch==0.1.7 \
    toolz==0.10.0 \
    tqdm==4.34.0 \
    typed_ast==1.4.0 \
    wandb==0.8.12 \
    wget==3.2\
    torch\
    transformers

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Open Ports for TensorBoard, Jupyter, and SSH
EXPOSE 6006
EXPOSE 7654
EXPOSE 22

WORKDIR /home/dev/src
COPY src/docs/THIRD_PARTY_NOTICE.md .

CMD bash
