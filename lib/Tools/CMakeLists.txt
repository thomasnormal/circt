add_subdirectory(arcilator)
add_subdirectory(circt-bmc)
# TODO: circt-debug needs fixes for C++17 compatibility
# add_subdirectory(circt-debug)
add_subdirectory(circt-lec)

# TODO: circt-verilog-lsp-server has slang API compatibility issues
# if(CIRCT_SLANG_FRONTEND_ENABLED)
#   add_subdirectory(circt-verilog-lsp-server)
# endif()
