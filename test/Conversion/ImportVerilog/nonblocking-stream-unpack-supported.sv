// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module NonBlockingStreamUnpackSupported;
  bit clk;
  bit arr[];
  int val;

  always @(posedge clk) begin
    { << bit { arr } } <= val;
  end
endmodule

// DIAG-NOT: non-blocking streaming unpack not yet supported

// IR-LABEL: moore.module @NonBlockingStreamUnpackSupported
// IR-LABEL: moore.procedure always
// IR: moore.wait_event {
// IR: moore.detect_event posedge
// IR: }
// IR-NOT: moore.wait_delay
// IR: %[[VAL:.+]] = moore.read %val : <i32>
// IR: moore.stream_unpack %arr, %[[VAL]] right_to_left true : <open_uarray<i1>>, i32
