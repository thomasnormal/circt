// CHECK-NOT: maxTime reached
// CHECK: DRV_CLK1 t=
// CHECK: DRV_RUN_START t=
//
// Regression: wait_event on virtual-interface field posedge inside func.call context
// must resume through an active call stack.
//
// RUN: circt-sim %s --max-time=200000000 2>&1 | FileCheck %s

module attributes {circt.rtti_parent_table = dense<0> : tensor<2xi32>} {
  llvm.mlir.global internal @"drv::__vtable__"(#llvm.zero) {addr_space = 0 : i32} : !llvm.array<1 x ptr>
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_format_time(i64) -> !llvm.struct<(ptr, i64)>
  func.func private @"drv::new"(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"drv", (i32, ptr, ptr)>
    llvm.store %arg1, %0 : !llvm.ptr, !llvm.ptr
    return
  }
  func.func private @"drv::run"(%arg0: !llvm.ptr) {
    %0 = sim.fmt.literal "DRV_RST_POS t="
    %1 = sim.fmt.literal "DRV_CLK2 t="
    %2 = sim.fmt.literal "DRV_CLK1 t="
    %3 = sim.fmt.literal "\0A"
    %4 = sim.fmt.literal "DRV_RUN_START t="
    %c1000000_i64 = hw.constant 1000000 : i64
    %5 = llhd.current_time
    %6 = llhd.time_to_int %5
    %7 = comb.divu %6, %c1000000_i64 : i64
    %8 = comb.mul %7, %c1000000_i64 : i64
    %9 = llvm.call @__moore_format_time(%8) : (i64) -> !llvm.struct<(ptr, i64)>
    %10 = sim.fmt.dyn_string %9 : !llvm.struct<(ptr, i64)>
    %11 = sim.fmt.concat (%4, %10, %3)
    sim.proc.print %11
    moore.wait_event {
      %33 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"drv", (i32, ptr, ptr)>
      %34 = llvm.load %33 : !llvm.ptr -> !llvm.ptr
      %35 = llvm.getelementptr %34[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.apb_if", (struct<(i1, i1)>, struct<(i1, i1)>, struct<(i1, i1)>)>
      %36 = llvm.load %35 : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %37 = llvm.extractvalue %36[0] : !llvm.struct<(i1, i1)> 
      %38 = llvm.extractvalue %36[1] : !llvm.struct<(i1, i1)> 
      %39 = hw.struct_create (%37, %38) : !hw.struct<value: i1, unknown: i1>
      %40 = builtin.unrealized_conversion_cast %39 : !hw.struct<value: i1, unknown: i1> to !moore.l1
      moore.detect_event posedge %40 : l1
    }
    %12 = llhd.current_time
    %13 = llhd.time_to_int %12
    %14 = comb.divu %13, %c1000000_i64 : i64
    %15 = comb.mul %14, %c1000000_i64 : i64
    %16 = llvm.call @__moore_format_time(%15) : (i64) -> !llvm.struct<(ptr, i64)>
    %17 = sim.fmt.dyn_string %16 : !llvm.struct<(ptr, i64)>
    %18 = sim.fmt.concat (%2, %17, %3)
    sim.proc.print %18
    moore.wait_event {
      %33 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"drv", (i32, ptr, ptr)>
      %34 = llvm.load %33 : !llvm.ptr -> !llvm.ptr
      %35 = llvm.getelementptr %34[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.apb_if", (struct<(i1, i1)>, struct<(i1, i1)>, struct<(i1, i1)>)>
      %36 = llvm.load %35 : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %37 = llvm.extractvalue %36[0] : !llvm.struct<(i1, i1)> 
      %38 = llvm.extractvalue %36[1] : !llvm.struct<(i1, i1)> 
      %39 = hw.struct_create (%37, %38) : !hw.struct<value: i1, unknown: i1>
      %40 = builtin.unrealized_conversion_cast %39 : !hw.struct<value: i1, unknown: i1> to !moore.l1
      moore.detect_event posedge %40 : l1
    }
    %19 = llhd.current_time
    %20 = llhd.time_to_int %19
    %21 = comb.divu %20, %c1000000_i64 : i64
    %22 = comb.mul %21, %c1000000_i64 : i64
    %23 = llvm.call @__moore_format_time(%22) : (i64) -> !llvm.struct<(ptr, i64)>
    %24 = sim.fmt.dyn_string %23 : !llvm.struct<(ptr, i64)>
    %25 = sim.fmt.concat (%1, %24, %3)
    sim.proc.print %25
    moore.wait_event {
      %33 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"drv", (i32, ptr, ptr)>
      %34 = llvm.load %33 : !llvm.ptr -> !llvm.ptr
      %35 = llvm.getelementptr %34[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.apb_if", (struct<(i1, i1)>, struct<(i1, i1)>, struct<(i1, i1)>)>
      %36 = llvm.load %35 : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %37 = llvm.extractvalue %36[0] : !llvm.struct<(i1, i1)> 
      %38 = llvm.extractvalue %36[1] : !llvm.struct<(i1, i1)> 
      %39 = hw.struct_create (%37, %38) : !hw.struct<value: i1, unknown: i1>
      %40 = builtin.unrealized_conversion_cast %39 : !hw.struct<value: i1, unknown: i1> to !moore.l1
      moore.detect_event posedge %40 : l1
    }
    %26 = llhd.current_time
    %27 = llhd.time_to_int %26
    %28 = comb.divu %27, %c1000000_i64 : i64
    %29 = comb.mul %28, %c1000000_i64 : i64
    %30 = llvm.call @__moore_format_time(%29) : (i64) -> !llvm.struct<(ptr, i64)>
    %31 = sim.fmt.dyn_string %30 : !llvm.struct<(ptr, i64)>
    %32 = sim.fmt.concat (%0, %31, %3)
    sim.proc.print %32
    return
  }
  hw.module @top() attributes {vpi.all_vars = {pclk = 1 : i32, preset_n = 1 : i32}, vpi.interface_defs = {apb_if = {pclk = 1 : i32, penable = 1 : i32, preset_n = 1 : i32}}, vpi.interface_instances = {i = "apb_if"}} {
    %0 = sim.fmt.literal "\0A"
    %1 = sim.fmt.literal "DONE t="
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.addressof @"drv::__vtable__" : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(20 : i64) : i64
    %7 = llhd.constant_time <0ns, 0d, 1e>
    %8 = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %9 = llvm.mlir.constant(6 : i64) : i64
    %c17000000_i64 = hw.constant 17000000 : i64
    %c5000000_i64 = hw.constant 5000000 : i64
    %c1000000_i64 = hw.constant 1000000 : i64
    %true = hw.constant true
    %10 = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
    %11 = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %12 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr
    %13 = llhd.sig %12 : !llvm.ptr
    %pclk = llhd.sig %11 : !hw.struct<value: i1, unknown: i1>
    %preset_n = llhd.sig %11 : !hw.struct<value: i1, unknown: i1>
    %14 = llhd.prb %pclk : !hw.struct<value: i1, unknown: i1>
    %15 = llhd.prb %13 : !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.apb_if", (struct<(i1, i1)>, struct<(i1, i1)>, struct<(i1, i1)>)>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %19 = llhd.prb %pclk : !hw.struct<value: i1, unknown: i1>
      %value = hw.struct_extract %19["value"] : !hw.struct<value: i1, unknown: i1>
      %20 = llvm.insertvalue %value, %8[0] : !llvm.struct<(i1, i1)> 
      %unknown = hw.struct_extract %19["unknown"] : !hw.struct<value: i1, unknown: i1>
      %21 = llvm.insertvalue %unknown, %20[1] : !llvm.struct<(i1, i1)> 
      llvm.store %21, %16 : !llvm.struct<(i1, i1)>, !llvm.ptr
      llhd.wait (%14 : !hw.struct<value: i1, unknown: i1>), ^bb1
    }
    %17 = llhd.prb %preset_n : !hw.struct<value: i1, unknown: i1>
    %18 = llvm.getelementptr %15[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.apb_if", (struct<(i1, i1)>, struct<(i1, i1)>, struct<(i1, i1)>)>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %19 = llhd.prb %preset_n : !hw.struct<value: i1, unknown: i1>
      %value = hw.struct_extract %19["value"] : !hw.struct<value: i1, unknown: i1>
      %20 = llvm.insertvalue %value, %8[0] : !llvm.struct<(i1, i1)> 
      %unknown = hw.struct_extract %19["unknown"] : !hw.struct<value: i1, unknown: i1>
      %21 = llvm.insertvalue %unknown, %20[1] : !llvm.struct<(i1, i1)> 
      llvm.store %21, %18 : !llvm.struct<(i1, i1)>, !llvm.ptr
      llhd.wait (%17 : !hw.struct<value: i1, unknown: i1>), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %19 = llhd.int_to_time %c5000000_i64
      llhd.wait delay %19, ^bb2
    ^bb2:  // pred: ^bb1
      %20 = llhd.prb %pclk : !hw.struct<value: i1, unknown: i1>
      %value = hw.struct_extract %20["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %20["unknown"] : !hw.struct<value: i1, unknown: i1>
      %21 = comb.xor %value, %true : i1
      %22 = comb.xor %unknown, %true : i1
      %23 = comb.and %21, %22 : i1
      %24 = hw.struct_create (%23, %unknown) : !hw.struct<value: i1, unknown: i1>
      llhd.drv %pclk, %24 after %7 : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb1
    }
    llhd.process {
      %19 = llhd.int_to_time %c17000000_i64
      llhd.wait delay %19, ^bb1
    ^bb1:  // pred: ^bb0
      llhd.drv %preset_n, %10 after %7 : !hw.struct<value: i1, unknown: i1>
      llhd.halt
    }
    llhd.process {
      %19 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
      llvm.store %5, %19 : i32, !llvm.ptr
      %20 = llvm.getelementptr %19[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"drv", (i32, ptr, ptr)>
      llvm.store %4, %20 : !llvm.ptr, !llvm.ptr
      %21 = llhd.prb %13 : !llvm.ptr
      func.call @"drv::new"(%19, %21) : (!llvm.ptr, !llvm.ptr) -> ()
      %22 = llvm.alloca %2 x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %3, %22 : !llvm.ptr, !llvm.ptr
      llvm.store %19, %22 : !llvm.ptr, !llvm.ptr
      %23 = llvm.load %22 : !llvm.ptr -> !llvm.ptr
      func.call @"drv::run"(%23) : (!llvm.ptr) -> ()
      %24 = llhd.current_time
      %25 = llhd.time_to_int %24
      %26 = comb.divu %25, %c1000000_i64 : i64
      %27 = comb.mul %26, %c1000000_i64 : i64
      %28 = llvm.call @__moore_format_time(%27) : (i64) -> !llvm.struct<(ptr, i64)>
      %29 = sim.fmt.dyn_string %28 : !llvm.struct<(ptr, i64)>
      %30 = sim.fmt.concat (%1, %29, %0)
      sim.proc.print %30
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
