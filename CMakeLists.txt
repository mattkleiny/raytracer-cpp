#
# Copyright 2018, the project authors. All rights reserved.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE.md file.
#

cmake_minimum_required(VERSION 3.10)

project(raytracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(raytracer raytracer.cpp)
target_link_libraries(raytracer png)