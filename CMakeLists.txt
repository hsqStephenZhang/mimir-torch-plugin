cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(torch)

if(NOT COMMAND add_mim_plugin)
    find_package(mim REQUIRED)
endif()

add_mim_plugin(torch
    SOURCES
        src/torch.cpp
        src/normalizers.cpp
        src/phase/lower.cpp
    INSTALL
)
