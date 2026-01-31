add_subdirectory(circt-debug)
add_subdirectory(circt-sim)

# Enable LSP server unit tests when slang is available
# Note: Requires gtest to be built - may not be available in all configurations
if(CIRCT_SLANG_FRONTEND_ENABLED AND TARGET llvm_gtest_main)
  add_subdirectory(circt-verilog-lsp-server)
endif()
