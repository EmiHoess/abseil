cmake_minimum_required(VERSION 3.25)

project(seil_consumer LANGUAGES CXX)

include(CTest)

find_package(seil CONFIG REQUIRED)

add_executable(seil_consumer main.cpp)
target_link_libraries(seil_consumer PRIVATE seil::seil)
add_test(NAME seil_consumer COMMAND seil_consumer)
