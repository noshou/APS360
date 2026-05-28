cmake_minimum_required(VERSION 3.15)
project(FormFact)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_program(OCAMLFIND ocamlfind REQUIRED)

set(FORMFACT_DIR ${CMAKE_SOURCE_DIR}/FormFact)
set(PARSERS_DIR  ${FORMFACT_DIR}/parsers)
set(DATA_DIR     ${FORMFACT_DIR}/data)

# --- Compile OCaml parsers ---

add_custom_command(
    OUTPUT  ${PARSERS_DIR}/F0Parser
    COMMAND ${OCAMLFIND} ocamlopt -package csv -linkpkg
                ${PARSERS_DIR}/F0Parser.ml -o ${PARSERS_DIR}/F0Parser
    DEPENDS ${PARSERS_DIR}/F0Parser.ml
    COMMENT "Compiling F0Parser"
)

add_custom_command(
    OUTPUT  ${PARSERS_DIR}/F1F2Parser
    COMMAND ${OCAMLFIND} ocamlopt -package str,yojson -linkpkg
                ${PARSERS_DIR}/F1F2Parser.ml -o ${PARSERS_DIR}/F1F2Parser
    DEPENDS ${PARSERS_DIR}/F1F2Parser.ml
    COMMENT "Compiling F1F2Parser"
)

# --- Run parsers to regenerate Fortran ---

add_custom_command(
    OUTPUT  ${FORMFACT_DIR}/F0Factor.f90
    COMMAND ${PARSERS_DIR}/F0Parser ${DATA_DIR}/f0_factors/f0.csv
    WORKING_DIRECTORY ${FORMFACT_DIR}
    DEPENDS ${PARSERS_DIR}/F0Parser
            ${DATA_DIR}/f0_factors/f0.csv
    COMMENT "Generating F0Factor.f90 from f0.csv"
)

add_custom_command(
    OUTPUT  ${FORMFACT_DIR}/F1F2Factors.f90
    COMMAND ${PARSERS_DIR}/F1F2Parser
                "${DATA_DIR}/f1_f2_factors/(12412.8)eV_f1_f2.json"
    WORKING_DIRECTORY ${FORMFACT_DIR}
    DEPENDS ${PARSERS_DIR}/F1F2Parser
            "${DATA_DIR}/f1_f2_factors/(12412.8)eV_f1_f2.json"
    COMMENT "Generating F1F2Factors.f90 from JSON"
    VERBATIM
)

# --- Compile FormFact Python extension with f2py ---

add_custom_target(formfact ALL
    COMMAND ${Python_EXECUTABLE} -m numpy.f2py -c
            ${FORMFACT_DIR}/formfact.pyf
            ${FORMFACT_DIR}/F0Factor.f90
            ${FORMFACT_DIR}/F1F2Factors.f90
            ${FORMFACT_DIR}/FormFact.f90
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${FORMFACT_DIR}/F0Factor.f90
            ${FORMFACT_DIR}/F1F2Factors.f90
            ${FORMFACT_DIR}/FormFact.f90
    COMMENT "Building FormFact Python extension with f2py"
)
