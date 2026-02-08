module JtagTargetDeviceDriverBfm;
  reg [31:0] registerBank [0:31];
  reg [4:0] instructionRegister;
  initial begin
    registerBank[instructionRegister] = registerBank[instructionRegister];
  end
endmodule
