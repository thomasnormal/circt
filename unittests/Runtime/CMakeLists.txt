add_circt_unittest(MooreRuntimeTests
  MooreRuntimeTest.cpp
)

target_link_libraries(MooreRuntimeTests
  PRIVATE
  MooreRuntime
)
