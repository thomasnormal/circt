set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(circt-mut circt-mut.cpp)
target_link_libraries(circt-mut
  PRIVATE
  CIRCTSupport
  LLVMSupport
)

llvm_update_compile_flags(circt-mut)
mlir_check_all_link_libraries(circt-mut)
