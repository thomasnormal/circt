// RUN: circt-sim %s | FileCheck %s

// CHECK: reg_val=0 reg_unk=0 out_val=0 out_unk=0

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c1_i64 = hw.constant 1000000 : i64
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_reg_val = sim.fmt.literal "reg_val="
  %fmt_reg_unk = sim.fmt.literal " reg_unk="
  %fmt_out_val = sim.fmt.literal " out_val="
  %fmt_out_unk = sim.fmt.literal " out_unk="
  %fmt_nl = sim.fmt.literal "\0A"

  %zero_struct = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
  %one_struct = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

  %clk = llhd.sig %false : i1
  %rst_ni = llhd.sig %zero_struct : !hw.struct<value: i1, unknown: i1>

  // Clock generator: 10ns period.
  llhd.process {
    llhd.drv %clk, %false after %eps : i1
    cf.br ^bb1
  ^bb1:
    %delay = llhd.int_to_time %c5_i64
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %clk_val = llhd.prb %clk : i1
    %clk_inv = comb.xor %clk_val, %true : i1
    llhd.drv %clk, %clk_inv after %eps : i1
    cf.br ^bb1
  }

  // Deassert reset at 10ns.
  llhd.process {
    %delay = llhd.int_to_time %c10_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %rst_ni, %one_struct after %eps : !hw.struct<value: i1, unknown: i1>
    llhd.halt
  }

  %rst_val = llhd.prb %rst_ni : !hw.struct<value: i1, unknown: i1>
  %rst_val_bit = hw.struct_extract %rst_val["value"] : !hw.struct<value: i1, unknown: i1>
  %rst_unk_bit = hw.struct_extract %rst_val["unknown"] : !hw.struct<value: i1, unknown: i1>
  %rst_unk_inv = comb.xor %rst_unk_bit, %true : i1
  %rst_known = comb.and bin %rst_val_bit, %rst_unk_inv : i1
  %rst_active = comb.xor %rst_known, %true : i1

  %clk_val = llhd.prb %clk : i1
  %clk_clock = seq.to_clock %clk_val
  %reg = seq.firreg %one_struct clock %clk_clock reset async %rst_active, %zero_struct preset 0 : !hw.struct<value: i1, unknown: i1>

  %out_sig = llhd.sig %zero_struct : !hw.struct<value: i1, unknown: i1>
  llhd.drv %out_sig, %reg after %eps : !hw.struct<value: i1, unknown: i1>

  // Print while reset is active.
  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %reg_val = hw.struct_extract %reg["value"] : !hw.struct<value: i1, unknown: i1>
    %reg_unk = hw.struct_extract %reg["unknown"] : !hw.struct<value: i1, unknown: i1>
    %out_probe = llhd.prb %out_sig : !hw.struct<value: i1, unknown: i1>
    %out_val = hw.struct_extract %out_probe["value"] : !hw.struct<value: i1, unknown: i1>
    %out_unk = hw.struct_extract %out_probe["unknown"] : !hw.struct<value: i1, unknown: i1>
    %fmt_reg_val_bits = sim.fmt.bin %reg_val : i1
    %fmt_reg_unk_bits = sim.fmt.bin %reg_unk : i1
    %fmt_out_val_bits = sim.fmt.bin %out_val : i1
    %fmt_out_unk_bits = sim.fmt.bin %out_unk : i1
    %fmt_out = sim.fmt.concat (%fmt_reg_val, %fmt_reg_val_bits, %fmt_reg_unk,
                               %fmt_reg_unk_bits, %fmt_out_val, %fmt_out_val_bits,
                               %fmt_out_unk, %fmt_out_unk_bits, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c20_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
