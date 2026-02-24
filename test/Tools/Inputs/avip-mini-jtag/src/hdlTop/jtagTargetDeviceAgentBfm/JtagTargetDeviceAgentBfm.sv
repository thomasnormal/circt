`timescale 1ns/1ps
module JtagTargetDeviceAgentBfm(JtagIf jtagIf);
  // Minimal reproduction of the AVIP bind that slang rejects if resolved in the
  // bound scope.
  JtagTargetDeviceMonitorBfm jtagTargetDeviceMonitorBfm (
    .clk(jtagIf.clk), .Tdi(jtagIf.Tdi), .Tdo(jtagIf.Tdo),
    .reset(jtagIf.reset), .Tms(jtagIf.Tms), .Trst(jtagIf.Trst)
  );

  bind JtagTargetDeviceMonitorBfm JtagTargetDeviceAssertions TestVectrorTestingAssertions(
    .clk(jtagIf.clk), .Tdo(jtagIf.Tdo), .Tms(jtagIf.Tms), .reset(jtagIf.reset)
  );
endmodule
