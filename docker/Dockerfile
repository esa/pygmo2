FROM quay.io/pypa/manylinux2014_x86_64:2022-10-25-fbea779

RUN yum install -y \
  eigen3-devel \
  coin-or-Ipopt-devel \
  boost169-devel boost169-graph boost169-serialization

# Newer version of TBB needed
RUN git clone https://github.com/oneapi-src/oneTBB.git && cd oneTBB && \
  git checkout v2021.7.0 && \
  mkdir build && cd build && \
  cmake -DCMAKE_CXX_FLAGS=-DTBB_ALLOCATOR_TRAITS_BROKEN .. && \
  make -j4 && \
  make install

# Newer version of NLopt-devel needed
RUN git clone https://github.com/stevengj/nlopt.git && cd nlopt && \
  mkdir build && cd build && \
  cmake .. && \
  make && \
  make install

RUN git clone https://github.com/esa/pagmo2.git && cd pagmo2 && \
  mkdir build && cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON \
    -DBOOST_INCLUDEDIR=/usr/include/boost169 -DBOOST_LIBRARYDIR=/usr/lib64/boost169 \
    -DPAGMO_WITH_EIGEN3=ON -DPAGMO_WITH_IPOPT=ON -DPAGMO_WITH_NLOPT=ON \
    -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DPAGMO_ENABLE_IPO=ON -DPAGMO_INSTALL_LIBDIR=lib && \
  make -j4 && \
  make install
