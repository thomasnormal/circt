// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module DPIOpenArraySupported;
  import "DPI-C" function void dpi_open_inout(
    input byte unsigned msg[],
    output byte unsigned digest[]
  );

  byte unsigned msg[];
  byte unsigned digest[];

  initial begin
    msg = new[2];
    msg[0] = 8'h12;
    msg[1] = 8'h34;
    digest = new[2];

    // Exercise call lowering with both input and output open-array arguments.
    dpi_open_inout(msg, digest);
    if (digest.size() != 2) begin
      $fatal(1, "unexpected digest size");
    end
  end

  // CHECK-DAG: func.func private @dpi_open_inout(!moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>)
  // CHECK: func.call @dpi_open_inout
endmodule

