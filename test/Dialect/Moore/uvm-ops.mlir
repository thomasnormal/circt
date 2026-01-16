// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

//===----------------------------------------------------------------------===//
// UVM Factory Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.uvm.type_register @MyComponent
moore.uvm.type_register @MyComponent @MyComponentClass component {
}

// CHECK-LABEL: moore.uvm.type_register @MyTransaction
moore.uvm.type_register @MyTransaction @MyTransactionClass object {
  // CHECK: moore.uvm.field_int @addr, 0
  moore.uvm.field_int @addr, 0
  // CHECK: moore.uvm.field_string @name, 0
  moore.uvm.field_string @name, 0
  // CHECK: moore.uvm.field_object @child, 1
  moore.uvm.field_object @child, 1
  // CHECK: moore.uvm.field_array_int @data_array, 0
  moore.uvm.field_array_int @data_array, 0
}

// CHECK-LABEL: moore.class.classdecl @TestTransaction {
moore.class.classdecl @TestTransaction {
  moore.class.propertydecl @data : !moore.i32
}

// CHECK-LABEL: moore.class.classdecl @TestComponent {
moore.class.classdecl @TestComponent {
  moore.class.propertydecl @name : !moore.string
}

// CHECK-LABEL: moore.module @TestFactoryOps
moore.module @TestFactoryOps() {
  moore.procedure initial {
    // CHECK: moore.uvm.create_object "my_transaction" : <@TestTransaction>
    %txn = moore.uvm.create_object "my_transaction" : !moore.class<@TestTransaction>

    // CHECK: moore.uvm.create_object "named_txn", "inst_name" : <@TestTransaction>
    %named = moore.uvm.create_object "named_txn", "inst_name" : !moore.class<@TestTransaction>

    // CHECK: moore.uvm.create_component "my_comp", "comp0", %{{.*}} : <@TestComponent> -> <@TestComponent>
    %parent = moore.class.new : <@TestComponent>
    %comp = moore.uvm.create_component "my_comp", "comp0", %parent : !moore.class<@TestComponent> -> !moore.class<@TestComponent>

    moore.return
  }
}

// CHECK-LABEL: moore.module @TestTypeOverrides
moore.module @TestTypeOverrides() {
  moore.procedure initial {
    // CHECK: moore.uvm.type_override "base_driver" -> "extended_driver"
    moore.uvm.type_override "base_driver" -> "extended_driver"

    // CHECK: moore.uvm.type_override "base_driver" -> "debug_driver" replace
    moore.uvm.type_override "base_driver" -> "debug_driver" replace

    // CHECK: moore.uvm.instance_override "base_driver" -> "debug_driver" at "env.agent.driver"
    moore.uvm.instance_override "base_driver" -> "debug_driver" at "env.agent.driver"

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// UVM Phase Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @TestPhaseClass {
moore.class.classdecl @TestPhaseClass {
  moore.class.propertydecl @data : !moore.i32
}

// CHECK-LABEL: moore.class.classdecl @TestUVMPhase {
moore.class.classdecl @TestUVMPhase {
  // Phase stub
}

// CHECK-LABEL: moore.module @TestPhaseOps
moore.module @TestPhaseOps() {
  moore.procedure initial {
    %phase = moore.class.new : <@TestUVMPhase>
    %count = moore.constant 1 : i32

    // CHECK: moore.uvm.raise_objection %{{.*}} : <@TestUVMPhase>
    moore.uvm.raise_objection %phase : !moore.class<@TestUVMPhase>

    // CHECK: moore.uvm.raise_objection %{{.*}}, %{{.*}} : <@TestUVMPhase>
    moore.uvm.raise_objection %phase, %count : !moore.class<@TestUVMPhase>

    // CHECK: moore.uvm.drop_objection %{{.*}} : <@TestUVMPhase>
    moore.uvm.drop_objection %phase : !moore.class<@TestUVMPhase>

    // CHECK: moore.uvm.drop_objection %{{.*}}, %{{.*}} : <@TestUVMPhase>
    moore.uvm.drop_objection %phase, %count : !moore.class<@TestUVMPhase>

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// UVM Config DB Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @TestConfigDbOps
moore.module @TestConfigDbOps() {
  moore.procedure initial {
    %comp = moore.class.new : <@TestComponent>
    %val = moore.constant 42 : i32

    // CHECK: moore.uvm.config_db.set "*.driver", "enable", %{{.*}} : !moore.i32
    moore.uvm.config_db.set "*.driver", "enable", %val : !moore.i32

    // CHECK: moore.uvm.config_db.set %{{.*}} : !moore.class<@TestComponent>, "uvm_test_top.*", "cfg", %{{.*}} : !moore.class<@TestComponent>
    moore.uvm.config_db.set %comp : !moore.class<@TestComponent>, "uvm_test_top.*", "cfg", %comp : !moore.class<@TestComponent>

    // CHECK: moore.uvm.config_db.get %{{.*}} : <@TestComponent>, "", "enable" -> !moore.i32
    %found, %retrieved = moore.uvm.config_db.get %comp : !moore.class<@TestComponent>, "", "enable" -> !moore.i32

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// UVM Sequence/TLM Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @TestSequence {
moore.class.classdecl @TestSequence {
  moore.class.propertydecl @id : !moore.i32
}

// CHECK-LABEL: moore.class.classdecl @TestSequencer {
moore.class.classdecl @TestSequencer {
  moore.class.propertydecl @id : !moore.i32
}

// CHECK-LABEL: moore.class.classdecl @TestItem {
moore.class.classdecl @TestItem {
  moore.class.propertydecl @data : !moore.i32
}

// CHECK-LABEL: moore.module @TestSequenceOps
moore.module @TestSequenceOps() {
  moore.procedure initial {
    %seq = moore.class.new : <@TestSequence>
    %sqr = moore.class.new : <@TestSequencer>
    %item = moore.class.new : <@TestItem>
    %parent_seq = moore.class.new : <@TestSequence>

    // CHECK: moore.uvm.sequence.start %{{.*}}, %{{.*}} : <@TestSequence>, <@TestSequencer>
    moore.uvm.sequence.start %seq, %sqr : !moore.class<@TestSequence>, !moore.class<@TestSequencer>

    // CHECK: moore.uvm.sequence.start %{{.*}}, %{{.*}}, parent %{{.*}} : !moore.class<@TestSequence> : <@TestSequence>, <@TestSequencer>
    moore.uvm.sequence.start %seq, %sqr, parent %parent_seq : !moore.class<@TestSequence> : !moore.class<@TestSequence>, !moore.class<@TestSequencer>

    // CHECK: moore.uvm.seq_item.start %{{.*}}, %{{.*}} : <@TestItem>, <@TestSequencer>
    moore.uvm.seq_item.start %item, %sqr : !moore.class<@TestItem>, !moore.class<@TestSequencer>

    // CHECK: moore.uvm.seq_item.finish %{{.*}}, %{{.*}} : <@TestItem>, <@TestSequencer>
    moore.uvm.seq_item.finish %item, %sqr : !moore.class<@TestItem>, !moore.class<@TestSequencer>

    // CHECK: moore.uvm.get_response %{{.*}} : <@TestSequencer> -> <@TestItem>
    %rsp = moore.uvm.get_response %sqr : !moore.class<@TestSequencer> -> !moore.class<@TestItem>

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// UVM TLM Port Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @TestPort {
moore.class.classdecl @TestPort {
  moore.class.propertydecl @id : !moore.i32
}

// CHECK-LABEL: moore.module @TestTLMOps
moore.module @TestTLMOps() {
  moore.procedure initial {
    %port = moore.class.new : <@TestPort>
    %txn = moore.class.new : <@TestTransaction>

    // CHECK: moore.uvm.tlm.put %{{.*}}, %{{.*}} : <@TestPort>, <@TestTransaction>
    moore.uvm.tlm.put %port, %txn : !moore.class<@TestPort>, !moore.class<@TestTransaction>

    // CHECK: moore.uvm.tlm.get %{{.*}} : <@TestPort> -> <@TestTransaction>
    %got = moore.uvm.tlm.get %port : !moore.class<@TestPort> -> !moore.class<@TestTransaction>

    // CHECK: moore.uvm.tlm.try_put %{{.*}}, %{{.*}} : <@TestPort>, <@TestTransaction>
    %put_ok = moore.uvm.tlm.try_put %port, %txn : !moore.class<@TestPort>, !moore.class<@TestTransaction>

    // CHECK: moore.uvm.tlm.try_get %{{.*}} : <@TestPort> -> <@TestTransaction>
    %get_ok, %try_got = moore.uvm.tlm.try_get %port : !moore.class<@TestPort> -> !moore.class<@TestTransaction>

    // CHECK: moore.uvm.tlm.analysis_write %{{.*}}, %{{.*}} : <@TestPort>, <@TestTransaction>
    moore.uvm.tlm.analysis_write %port, %txn : !moore.class<@TestPort>, !moore.class<@TestTransaction>

    moore.return
  }
}

// UVM Reporting operations require a !moore.string which is dynamic -
// testing skipped for now until we have proper string conversion ops.

//===----------------------------------------------------------------------===//
// UVM Base Class Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.uvm.class.decl @MyDriver extends uvm_driver {
moore.uvm.class.decl @MyDriver extends uvm_driver {
}

// CHECK-LABEL: moore.uvm.class.decl @MyMonitor extends uvm_monitor from @CustomMonitorBase {
moore.uvm.class.decl @MyMonitor extends uvm_monitor from @CustomMonitorBase {
}

// CHECK-LABEL: moore.module @TestHierarchyOps
moore.module @TestHierarchyOps() {
  moore.procedure initial {
    %comp = moore.class.new : <@TestComponent>

    // CHECK: moore.uvm.get_parent %{{.*}} : <@TestComponent> -> <@TestComponent>
    %parent = moore.uvm.get_parent %comp : !moore.class<@TestComponent> -> !moore.class<@TestComponent>

    // CHECK: moore.uvm.get_full_name %{{.*}} : <@TestComponent>
    %name = moore.uvm.get_full_name %comp : !moore.class<@TestComponent>

    // CHECK: moore.uvm.find_by_name "env.agent.driver" -> <@TestComponent>
    %found = moore.uvm.find_by_name "env.agent.driver" -> !moore.class<@TestComponent>

    // CHECK: moore.uvm.find_by_name "agent.driver", %{{.*}} : !moore.class<@TestComponent> -> <@TestComponent>
    %found2 = moore.uvm.find_by_name "agent.driver", %comp : !moore.class<@TestComponent> -> !moore.class<@TestComponent>

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// UVM TLM Port Declaration and Connection
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @TLMDriver {
moore.class.classdecl @TLMDriver {
  moore.class.propertydecl @seq_item_port : !moore.class<@TestPort>
}

// CHECK-LABEL: moore.class.classdecl @TLMSequencer {
moore.class.classdecl @TLMSequencer {
  moore.class.propertydecl @seq_item_export : !moore.class<@TestPort>
}

// CHECK-LABEL: moore.module @TestTLMConnections
moore.module @TestTLMConnections() {
  moore.procedure initial {
    %driver = moore.class.new : <@TLMDriver>
    %sqr = moore.class.new : <@TLMSequencer>
    %monitor = moore.class.new : <@TestComponent>
    %scoreboard = moore.class.new : <@TestComponent>

    // CHECK: moore.uvm.tlm.connect %{{.*}}, %{{.*}} : <@TLMDriver>, <@TLMSequencer>
    moore.uvm.tlm.connect %driver, %sqr : !moore.class<@TLMDriver>, !moore.class<@TLMSequencer>

    // CHECK: moore.uvm.analysis.connect %{{.*}}, %{{.*}} : <@TestComponent>, <@TestComponent>
    moore.uvm.analysis.connect %monitor, %scoreboard : !moore.class<@TestComponent>, !moore.class<@TestComponent>

    moore.return
  }
}
