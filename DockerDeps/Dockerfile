FROM ubuntu:21.10
WORKDIR /tmp/docker_build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && \
    apt-get -y upgrade && \
    apt -y install clang-13 cython3 python3-pip libopenmpi-dev cmake gcc-11 libgsl-dev libfftw3-3 libfftw3-dev libsuitesparse-dev git openssh-server sudo && \
    pip3 install --upgrade pip && \
    pip3 install tensorflow-gpu scipy greenlet
RUN git clone https://github.com/UIUC-PPL/charm4py && \
    cd charm4py && \
    sed -i -E -e "s/(\s+)(args\ \+\=)/\1args\.remove('\+\+interactive')\n\1\2/" ./charmrun/start.py && \
    git clone https://github.com/UIUC-PPL/charm charm_src/charm
WORKDIR /tmp/docker_build
# Build Fast BATLLNN dependencies
RUN wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz && \
    tar -xf glpk-5.0.tar.gz && \
    cd glpk-5.0 && \
    CC=/usr/bin/clang-13 CXX=/usr/bin/clang++-13 CPP=/usr/bin/clang-cpp-13 ./configure --prefix /usr/local && make && make install && \
    cd .. && \
    git clone https://github.com/cvxopt/cvxopt.git && \
    cd cvxopt && \
    CVXOPT_BUILD_FFTW=1 CVXOPT_BUILD_GSL=1 CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR=/usr/local/lib CVXOPT_GLPK_INC_DIR=/usr/local/include CC=/usr/bin/clang-13 CXX=/usr/bin/clang++-13 CPP=/usr/bin/clang-cpp-13 python3 setup.py install --user && \
    cd .. && \
    wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew && \
    chmod 755 coinbrew && \
    ./coinbrew --latest-release --no-prompt build Clp --prefix=/usr/local ADD_FFLAGS=-fallow-argument-mismatch && \
    ./coinbrew --latest-release --no-prompt build Cbc --prefix=/usr/local ADD_FFLAGS=-fallow-argument-mismatch && \
    COIN_INSTALL_DIR=/usr/local && \
    COIN_INSTALL_DIR=/usr/local pip3 install cylp && \
    cd charm4py && \
    python3 setup.py install --mpi && \
    cd .. && \
    rm -rf *
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
RUN echo 'ubuntu:ubuntu' | chpasswd
RUN service ssh start
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
