cmake_minimum_required(VERSION 3.25)

project(abseil_consumer LANGUAGES CXX)

include(CTest)

find_package(abseil CONFIG REQUIRED)

add_executable(abseil_consumer main.cpp)
target_link_libraries(abseil_consumer PRIVATE abseil::abseil)
add_test(NAME abseil_consumer COMMAND abseil_consumer)
