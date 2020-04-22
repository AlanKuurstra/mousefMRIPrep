#FROM ubuntu:xenial
FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    wget \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config \
                    git \
                    python3 \
                    python3-dev \
                    python3-pip \
                    && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y --no-install-recommends \
                    nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# set locale - this is neccsary at least for python's defaults in opening certain files from disk
RUN apt-get update && apt-get install -y --no-install-recommends locales
RUN sed -i -e 's/# \(en_US\.UTF-8 .*\)/\1/' /etc/locale.gen && \
sed -i -e 's/# \(en_CA\.UTF-8 .*\)/\1/' /etc/locale.gen && locale-gen
ENV LANG en_CA.UTF-8
ENV LANGUAGE en_CA:en
ENV LC_ALL en_CA.UTF-8

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users mousefmriprep
ENV HOME="/home/mousefmriprep"

#=============================================
#install BrainSuite18a
#=============================================
RUN wget -q http://brainsuite.org/data/BIDS/BrainSuite18a.BIDS.tgz && \
    tar -xf BrainSuite18a.BIDS.tgz && \
    mv /BrainSuite18a /opt && \
    cd /opt/BrainSuite18a/bin && \
    chmod -R ugo+r /opt/BrainSuite18a && \
    cd / && \
    rm BrainSuite18a.BIDS.tgz
RUN chmod +x /opt/BrainSuite18a
RUN chmod -R +x /opt/BrainSuite18a/bin/
RUN chmod -R +x /opt/BrainSuite18a/svreg/bin/
RUN chmod -R +x /opt/BrainSuite18a/bdp/
ENV PATH=/opt/BrainSuite18a/bin/:/opt/BrainSuite18a/svreg/bin/:/opt/BrainSuite18a/bdp/:${PATH}
#=============================================

#=============================================
#install Neurodebian repo
#=============================================
#RUN echo "America/New_York" | sudo tee /etc/timezone && sudo dpkg-reconfigure --frontend noninteractive tzdata
#RUN wget -O- http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
apt-key adv --recv-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9
#RUN apt-get update
#=============================================

#=============================================
#install fsl
#=============================================
RUN apt-get update && apt-get install -y fsl=5.0.9-5~nd18.04+1 && \
apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV FSLDIR=/usr/share/fsl/5.0 \
    POSSUMDIR=$FSLDIR \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    FSLBROWSER=/etc/alternatives/x-www-browser \
    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:${LD_LIBRARY_PATH} \
    PATH=/usr/lib/fsl/5.0:$PATH
#=============================================

#=============================================
#install afni
#=============================================
RUN apt-get update && apt-get install -y afni=18.0.05+git24-gb25b21054~dfsg.1-1~nd17.10+1+nd18.04+1 && \
apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV AFNI_MODELPATH="/usr/lib/afni/models" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_TTATLAS_DATASET="/usr/share/afni/atlases" \
    AFNI_PLUGINPATH="/usr/lib/afni/plugins" \
    PATH="/usr/lib/afni/bin:$PATH"
#=============================================

RUN apt-get update && apt-get install -y --no-install-recommends heudiconv && \
apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#=============================================
#install ANTs 2.1.0 (NeuroDocker build) (fails on anat to atlas registration)
#=============================================
#ENV ANTSPATH=/usr/lib/ants
#RUN mkdir -p $ANTSPATH && \
#    curl -sSL "https://dl.dropbox.com/s/h8k4v6d1xrv0wbe/ANTs-Linux-centos5_x86_64-v2.1.0-78931aa.tar.gz" \
#    | tar -xzC $ANTSPATH --strip-components 1
#ENV PATH=$ANTSPATH:$PATH
#=============================================
# install ANTs 2.1.0-gGIT-N (apt build, note: apt says 2.2.0 but antsRegistration --version reports different)
#=============================================
RUN apt-get update && apt-get install -y ants=2.2.0-1ubuntu1 && \
apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV ANTSPATH=/usr/lib/ants
ENV PATH=$ANTSPATH:$PATH
# consider upgrading to ANTs 2.3.0 with NeuroDocker builds, would have to chnage the masks connections
#=============================================

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

#=============================================
#install pipeline's python dependencies
#=============================================
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy==1.16.2 setuptools>=40.8.0
COPY requirements.txt /code/
RUN python3 -m pip install -r /code/requirements.txt
#=============================================

COPY . /code/
RUN chmod +x /code/tar2bids.py
RUN chmod +x /code/fix_mouse_bids.py
ENV PATH=/code:$PATH

WORKDIR /code
RUN chmod +x run.py
ENTRYPOINT ["/code/run.py"]

