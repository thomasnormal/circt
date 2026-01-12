add_subdirectory(arcilator)
add_subdirectory(circt-bmc)
add_subdirectory(circt-debug)
add_subdirectory(circt-lec)

# TODO: circt-verilog-lsp-server has slang API compatibility issues
# if(CIRCT_SLANG_FRONTEND_ENABLED)
#   add_subdirectory(circt-verilog-lsp-server)
# endif()
