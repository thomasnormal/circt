# MooreRuntime - Runtime library for compiled Moore simulations
#
# This library provides runtime support for operations that cannot be
# lowered directly to LLVM IR during MooreToCore conversion.

add_library(MooreRuntime STATIC
  MooreRuntime.cpp
)

target_include_directories(MooreRuntime PUBLIC
  ${CIRCT_SOURCE_DIR}/include
)

# This is a standalone runtime library that will be linked with
# compiled simulations, so it should not depend on MLIR/LLVM libs.

set_target_properties(MooreRuntime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Enable exceptions for regex error handling
target_compile_options(MooreRuntime PRIVATE -fexceptions)

# Install the runtime library and header for users who compile simulations
install(TARGETS MooreRuntime
  ARCHIVE DESTINATION lib
)

install(FILES
  ${CIRCT_SOURCE_DIR}/include/circt/Runtime/MooreRuntime.h
  DESTINATION include/circt/Runtime
)
