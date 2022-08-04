cmake_minimum_required(VERSION 3.22)
project(allreduce C)

set(CMAKE_C_STANDARD 11)

include_directories(src)

add_executable(allreduce
        src/allreduce.c
        src/allreduce_client.c
        src/allreduce_client.h
        src/allreduce_core.c
        src/allreduce_core.h
        src/allreduce_daemon.c
        src/allreduce_daemon.h
        src/allreduce_ucx.c
        src/allreduce_ucx.h)
