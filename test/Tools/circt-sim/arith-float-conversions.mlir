// RUN: circt-sim %s | FileCheck %s

// CHECK: q=2
// CHECK: s=1
// CHECK: z=0
// CHECK: cmp=1
// CHECK: h=2

hw.module @test() {
  %fmt_q = sim.fmt.literal "q="
  %fmt_s = sim.fmt.literal "s="
  %fmt_z = sim.fmt.literal "z="
  %fmt_cmp = sim.fmt.literal "cmp="
  %fmt_h = sim.fmt.literal "h="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    cf.br ^entry
  ^entry:
    %c10_i32 = arith.constant 10 : i32
    %c4_i32 = arith.constant 4 : i32
    %cneg3_i32 = arith.constant -3 : i32
    %fhalf = arith.constant 5.000000e-01 : f64

    %f10 = arith.uitofp %c10_i32 : i32 to f64
    %f4 = arith.uitofp %c4_i32 : i32 to f64
    %fneg3 = arith.sitofp %cneg3_i32 : i32 to f64

    %qf = arith.divf %f10, %f4 : f64
    %q = arith.fptoui %qf : f64 to i32

    %sf = arith.addf %fneg3, %f4 : f64
    %s = arith.fptosi %sf : f64 to i32

    %mf = arith.mulf %sf, %qf : f64
    %zf = arith.subf %mf, %qf : f64
    %z = arith.fptosi %zf : f64 to i32

    %cmp = arith.cmpf ogt, %qf, %sf : f64

    %hf = arith.divf %sf, %fhalf : f64
    %h = arith.fptoui %hf : f64 to i32

    %fmt_q_val = sim.fmt.dec %q : i32
    %fmt_q_out = sim.fmt.concat (%fmt_q, %fmt_q_val, %fmt_nl)
    sim.proc.print %fmt_q_out

    %fmt_s_val = sim.fmt.dec %s : i32
    %fmt_s_out = sim.fmt.concat (%fmt_s, %fmt_s_val, %fmt_nl)
    sim.proc.print %fmt_s_out

    %fmt_z_val = sim.fmt.dec %z : i32
    %fmt_z_out = sim.fmt.concat (%fmt_z, %fmt_z_val, %fmt_nl)
    sim.proc.print %fmt_z_out

    %fmt_cmp_val = sim.fmt.dec %cmp : i1
    %fmt_cmp_out = sim.fmt.concat (%fmt_cmp, %fmt_cmp_val, %fmt_nl)
    sim.proc.print %fmt_cmp_out

    %fmt_h_val = sim.fmt.dec %h : i32
    %fmt_h_out = sim.fmt.concat (%fmt_h, %fmt_h_val, %fmt_nl)
    sim.proc.print %fmt_h_out

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
