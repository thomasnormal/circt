// RUN: circt-verilog --ir-hw --top=repro_aes_dom %s | FileCheck %s --check-prefix=HW
// RUN: circt-verilog --ir-moore --top=repro_aes_dom %s | FileCheck %s --check-prefix=MOORE

module repro_aes_dom #(
  parameter bit SecMasking = 1
) (
  input logic mux_sel_err_i,
  input logic sp_enc_err_i,
  input logic op_err_i,
  input logic prng_reseed_q_i,
  input logic prng_reseed_done_q,
  input logic advance,
  output logic out_valid_o
);
  always_comb begin
    out_valid_o = (mux_sel_err_i || sp_enc_err_i || op_err_i) ? 1'b0 :
        SecMasking ? (prng_reseed_q_i ? (prng_reseed_done_q & advance) : advance)
                   : advance;
  end
endmodule

// HW: hw.module @repro_aes_dom
// MOORE-LABEL: moore.module @repro_aes_dom
// MOORE: moore.procedure always_comb {
// MOORE: %[[ERR0:.+]] = moore.conditional %{{.+}} : l1 -> l1
// MOORE: %[[ERR1:.+]] = moore.conditional %[[ERR0]] : l1 -> l1
// MOORE: %[[SEL:.+]] = moore.conditional %[[ERR1]] : l1 -> l1
// MOORE: %[[MASKED:.+]] = moore.conditional %{{.+}} : i1 -> l1
// MOORE: %[[RESEED:.+]] = moore.conditional %{{.+}} : l1 -> l1
// MOORE: %[[DONE:.+]] = moore.read %{{.+}} : <l1>
// MOORE: %[[ADV:.+]] = moore.read %{{.+}} : <l1>
// MOORE: %[[GATED:.+]] = moore.and %[[DONE]], %[[ADV]] : l1
// MOORE: moore.blocking_assign %out_valid_o, %[[SEL]] : l1
