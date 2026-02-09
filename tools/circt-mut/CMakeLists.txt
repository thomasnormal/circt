set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(circt-mut circt-mut.cpp)
target_link_libraries(circt-mut
  PRIVATE
  CIRCTSupport
  LLVMSupport
)

llvm_update_compile_flags(circt-mut)
mlir_check_all_link_libraries(circt-mut)

install(PROGRAMS
  "${CIRCT_SOURCE_DIR}/utils/run_mutation_cover.sh"
  "${CIRCT_SOURCE_DIR}/utils/run_mutation_matrix.sh"
  "${CIRCT_SOURCE_DIR}/utils/generate_mutations_yosys.sh"
  "${CIRCT_SOURCE_DIR}/utils/create_mutated_yosys.sh"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/circt/utils"
  COMPONENT circt-mut
)
