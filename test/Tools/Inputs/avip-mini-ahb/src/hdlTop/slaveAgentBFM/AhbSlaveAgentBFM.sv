module AhbSlaveAgentBFM(AhbInterface ahbInterface);
  bind AhbSlaveMonitorBFM AhbSlaveAssertion ahb_assert (.hclk(ahbInterface.hclk),
                                                         .hready(ahbInterface.hready));
  bind AhbSlaveMonitorBFM AhbSlaveCoverProperty ahb_cover (.hclk(ahbInterface.hclk),
                                                            .hready(ahbInterface.hready));
endmodule
