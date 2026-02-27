// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s --check-prefix=OUT
// REQUIRES: slang

module top;
  bit config, library, cell, design, endconfig;
  bit include, incdir, instance, liblist, use;

  initial begin
    config = 1'b1;
    library = 1'b0;
    cell = 1'b1;
    design = 1'b0;
    endconfig = 1'b1;
    include = 1'b1;
    incdir = 1'b0;
    instance = 1'b1;
    liblist = 1'b0;
    use = 1'b1;

    if ({config, library, cell, design, endconfig,
         include, incdir, instance, liblist, use} !== 10'b1010110101) begin
      $display("FAIL kw=%b", {config, library, cell, design, endconfig,
                              include, incdir, instance, liblist, use});
      $fatal(1);
    end

    $display("PASS kw=%b", {config, library, cell, design, endconfig,
                            include, incdir, instance, liblist, use});
    $finish;
  end
endmodule

// OUT: PASS kw=1010110101
