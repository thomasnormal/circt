module JtagControllerDeviceAgentBfm;
  bind jtagControllerDeviceMonitorBfm JtagControllerDeviceAssertions
      TestVectrorTestingAssertions(.clk(clk), .Tdi(Tdi), .reset(reset), .Tms(Tms));
endmodule
