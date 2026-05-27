cmake_minimum_required(VERSION 3.15)
project(FormFact)

find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB FORMFACT_SOURCES "${CMAKE_SOURCE_DIR}/Molecule/FormFact/*.f90")

add_custom_target(formfact ALL
    COMMAND ${Python_EXECUTABLE} -m numpy.f2py -c
            "${CMAKE_SOURCE_DIR}/formfact.pyf"
            ${FORMFACT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building FormFact extension with f2py"
)
