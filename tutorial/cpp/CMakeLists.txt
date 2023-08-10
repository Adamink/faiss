# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.71.0 COMPONENTS program_options) 

add_executable(1-Flat EXCLUDE_FROM_ALL 1-Flat.cpp)
target_link_libraries(1-Flat PRIVATE faiss)

add_executable(2-IVFFlat EXCLUDE_FROM_ALL 2-IVFFlat.cpp)
target_link_libraries(2-IVFFlat PRIVATE faiss Boost::program_options)

add_executable(3-IVFPQ EXCLUDE_FROM_ALL 3-IVFPQ.cpp)
target_link_libraries(3-IVFPQ PRIVATE faiss)

add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
target_link_libraries(4-GPU PRIVATE faiss Boost::program_options)

add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
target_link_libraries(5-Multiple-GPUs PRIVATE faiss)

add_executable(IVFPQ-GPU EXCLUDE_FROM_ALL IVFPQ-GPU.cpp)
target_link_libraries(IVFPQ-GPU PRIVATE faiss Boost::program_options)

add_executable(IVFFlat-GPU EXCLUDE_FROM_ALL IVFFlat-GPU.cpp)
target_link_libraries(IVFFlat-GPU PRIVATE faiss Boost::program_options)

add_executable(IVFPQ-CPU EXCLUDE_FROM_ALL IVFPQ-CPU.cpp)
target_link_libraries(IVFFlat-GPU PRIVATE faiss Boost::program_options)