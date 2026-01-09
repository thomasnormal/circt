include_directories(${CIRCT_MAIN_SRC_DIR})
add_circt_unittest(CIRCTSimTests
  DesignPartitionerTest.cpp
  EventQueueTest.cpp
  ForkJoinTest.cpp
  IncrementalCompilerTest.cpp
  MemoryOptimizationTest.cpp
  ParallelSchedulerTest.cpp
  PerformanceProfilerTest.cpp
  ProcessSchedulerTest.cpp
  SimulationControlTest.cpp
  SyncPrimitivesTest.cpp
  TLMPortsTest.cpp
  UVMFactoryTest.cpp
  UVMPhaseManagerTest.cpp
)
target_link_libraries(CIRCTSimTests PRIVATE CIRCTSim)
