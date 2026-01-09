set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(circt-cov circt-cov.cpp)
target_link_libraries(circt-cov
  PRIVATE
  CIRCTSupport
  LLVMSupport
)

llvm_update_compile_flags(circt-cov)
mlir_check_all_link_libraries(circt-cov)
