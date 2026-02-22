module AhbMasterAgentBFM(AhbInterface ahbInterface);
  bind AhbMasterMonitorBFM AhbMasterAssertion ahb_assert (.hclk(ahbInterface.hclk),
                                                           .hready(ahbInterface.hready));
  bind AhbMasterMonitorBFM AhbMasterCoverProperty ahb_cover (.hclk(ahbInterface.hclk),
                                                              .hready(ahbInterface.hready));
endmodule
