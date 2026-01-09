include_directories(${CIRCT_MAIN_SRC_DIR})
add_circt_unittest(CIRCTSimTests
  EventQueueTest.cpp
  ProcessSchedulerTest.cpp
  ForkJoinTest.cpp
  SyncPrimitivesTest.cpp
  UVMPhaseManagerTest.cpp
  UVMFactoryTest.cpp
  TLMPortsTest.cpp
  SimulationControlTest.cpp
)
target_link_libraries(CIRCTSimTests PRIVATE CIRCTSim)
