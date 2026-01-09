include_directories(${CIRCT_MAIN_SRC_DIR})
add_circt_unittest(CIRCTSimTests
  EventQueueTest.cpp
  ProcessSchedulerTest.cpp
)
target_link_libraries(CIRCTSimTests PRIVATE CIRCTSim)
