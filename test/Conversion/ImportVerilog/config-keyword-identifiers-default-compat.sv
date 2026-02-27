// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module ConfigKeywordIdentifiersDefaultCompat;
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
    $display("KW=%b", {config, library, cell, design, endconfig,
                       include, incdir, instance, liblist, use});
  end
endmodule

// IR-LABEL: moore.module @ConfigKeywordIdentifiersDefaultCompat
// IR: moore.builtin.display
