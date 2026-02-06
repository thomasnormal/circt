// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%false = hw.constant false
%clk = hw.constant true

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !sva.sequence
// CHECK: unrealized_conversion_cast to !sva.property
%s = unrealized_conversion_cast to !sva.sequence
%p = unrealized_conversion_cast to !sva.property

//===----------------------------------------------------------------------===//
// Sequence Operations
//===----------------------------------------------------------------------===//

// CHECK: sva.seq.delay {{%.+}}, 0 : i1
// CHECK: sva.seq.delay {{%.+}}, 2 : i1
// CHECK: sva.seq.delay {{%.+}}, 1, 2 : !sva.sequence
sva.seq.delay %true, 0 : i1
sva.seq.delay %true, 2 : i1
sva.seq.delay %s, 1, 2 : !sva.sequence

// CHECK: sva.seq.repeat {{%.+}}, 3 : i1
// CHECK: sva.seq.repeat {{%.+}}, 2, 2 : !sva.sequence
sva.seq.repeat %true, 3 : i1
sva.seq.repeat %s, 2, 2 : !sva.sequence

// CHECK: sva.seq.goto_repeat {{%.+}}, 2, 1 : !sva.sequence
sva.seq.goto_repeat %s, 2, 1 : !sva.sequence
// CHECK: sva.seq.goto_repeat {{%.+}}, 2 : !sva.sequence
sva.seq.goto_repeat %s, 2 : !sva.sequence

// CHECK: sva.seq.non_consecutive_repeat {{%.+}}, 2, 1 : !sva.sequence
sva.seq.non_consecutive_repeat %s, 2, 1 : !sva.sequence
// CHECK: sva.seq.non_consecutive_repeat {{%.+}}, 2 : !sva.sequence
sva.seq.non_consecutive_repeat %s, 2 : !sva.sequence

// CHECK: sva.seq.concat {{%.+}}, {{%.+}} : !sva.sequence, !sva.sequence
sva.seq.concat %s, %s : !sva.sequence, !sva.sequence

// CHECK: sva.seq.or {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.seq.or {{%.+}}, {{%.+}} : !sva.sequence, !sva.sequence
sva.seq.or %true, %false : i1, i1
sva.seq.or %s, %s : !sva.sequence, !sva.sequence

// CHECK: sva.seq.and {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.seq.and {{%.+}}, {{%.+}} : !sva.sequence, !sva.sequence
sva.seq.and %true, %false : i1, i1
sva.seq.and %s, %s : !sva.sequence, !sva.sequence

// CHECK: sva.seq.intersect {{%.+}}, {{%.+}} : !sva.sequence, !sva.sequence
sva.seq.intersect %s, %s : !sva.sequence, !sva.sequence

// CHECK: sva.seq.first_match {{%.+}} : !sva.sequence
sva.seq.first_match %s : !sva.sequence

// CHECK: sva.seq.within {{%.+}}, {{%.+}} : !sva.sequence, !sva.sequence
sva.seq.within %s, %s : !sva.sequence, !sva.sequence

// CHECK: sva.seq.throughout {{%.+}}, {{%.+}} : i1, !sva.sequence
sva.seq.throughout %true, %s : i1, !sva.sequence

// CHECK: sva.seq.clock {{%.+}}, posedge {{%.+}} : !sva.sequence
// CHECK: sva.seq.clock {{%.+}}, negedge {{%.+}} : !sva.sequence
// CHECK: sva.seq.clock {{%.+}}, edge {{%.+}} : i1
sva.seq.clock %s, posedge %clk : !sva.sequence
sva.seq.clock %s, negedge %clk : !sva.sequence
sva.seq.clock %true, edge %clk : i1

//===----------------------------------------------------------------------===//
// Property Operations
//===----------------------------------------------------------------------===//

// CHECK: sva.prop.not {{%.+}} : i1
// CHECK: sva.prop.not {{%.+}} : !sva.sequence
// CHECK: sva.prop.not {{%.+}} : !sva.property
sva.prop.not %true : i1
sva.prop.not %s : !sva.sequence
sva.prop.not %p : !sva.property

// CHECK: sva.prop.and {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.prop.and {{%.+}}, {{%.+}} : !sva.property, !sva.property
sva.prop.and %true, %false : i1, i1
sva.prop.and %p, %p : !sva.property, !sva.property

// CHECK: sva.prop.or {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.prop.or {{%.+}}, {{%.+}} : !sva.property, !sva.property
sva.prop.or %true, %false : i1, i1
sva.prop.or %p, %p : !sva.property, !sva.property

// CHECK: sva.prop.implication {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.prop.implication {{%.+}}, {{%.+}} overlapping : i1, i1
// CHECK: sva.prop.implication {{%.+}}, {{%.+}} : !sva.sequence, !sva.property
sva.prop.implication %true, %true : i1, i1
sva.prop.implication %true, %true overlapping : i1, i1
sva.prop.implication %s, %p : !sva.sequence, !sva.property

// CHECK: sva.prop.if {{%.+}}, {{%.+}} : i1, !sva.property
// CHECK: sva.prop.if {{%.+}}, {{%.+}}, {{%.+}} : i1, !sva.property, !sva.property
sva.prop.if %true, %p : i1, !sva.property
sva.prop.if %true, %p, %p : i1, !sva.property, !sva.property

// CHECK: sva.prop.eventually {{%.+}} : i1
// CHECK: sva.prop.eventually {{%.+}} : !sva.property
sva.prop.eventually %true : i1
sva.prop.eventually %p : !sva.property

// CHECK: sva.prop.always {{%.+}} : i1
// CHECK: sva.prop.always {{%.+}} : !sva.property
sva.prop.always %true : i1
sva.prop.always %p : !sva.property

// CHECK: sva.prop.until {{%.+}}, {{%.+}} : i1, i1
// CHECK: sva.prop.until {{%.+}}, {{%.+}} strong : !sva.property, !sva.property
sva.prop.until %true, %false : i1, i1
sva.prop.until %p, %p strong : !sva.property, !sva.property

// CHECK: sva.prop.nexttime {{%.+}} : i1
// CHECK: sva.prop.nexttime {{%.+}}, 2 : !sva.property
sva.prop.nexttime %true : i1
sva.prop.nexttime %p, 2 : !sva.property

// CHECK: sva.prop.clock {{%.+}}, posedge {{%.+}} : !sva.property
sva.prop.clock %p, posedge %clk : !sva.property

//===----------------------------------------------------------------------===//
// Disable Condition
//===----------------------------------------------------------------------===//

// CHECK: sva.disable_iff {{%.+}}, {{%.+}} : i1, !sva.property
sva.disable_iff %true, %p : i1, !sva.property

//===----------------------------------------------------------------------===//
// Assertion Directives
//===----------------------------------------------------------------------===//

// CHECK: sva.assert {{%.+}} : i1
// CHECK: sva.assert {{%.+}} label "test_assert" : !sva.property
// CHECK: sva.assert {{%.+}} if {{%.+}} label "cond_assert" message "failed!" : !sva.property
sva.assert %true : i1
sva.assert %p label "test_assert" : !sva.property
sva.assert %p if %true label "cond_assert" message "failed!" : !sva.property

// CHECK: sva.assume {{%.+}} : i1
// CHECK: sva.assume {{%.+}} label "test_assume" : !sva.property
sva.assume %true : i1
sva.assume %p label "test_assume" : !sva.property

// CHECK: sva.cover {{%.+}} : i1
// CHECK: sva.cover {{%.+}} label "test_cover" : !sva.property
sva.cover %true : i1
sva.cover %p label "test_cover" : !sva.property

// CHECK: sva.expect {{%.+}} : i1
sva.expect %true : i1

//===----------------------------------------------------------------------===//
// Clocked Assertion Directives
//===----------------------------------------------------------------------===//

// CHECK: sva.clocked_assert {{%.+}}, posedge {{%.+}} : !sva.property
// CHECK: sva.clocked_assert {{%.+}}, posedge {{%.+}} label "clocked_test" : !sva.property
sva.clocked_assert %p, posedge %clk : !sva.property
sva.clocked_assert %p, posedge %clk label "clocked_test" : !sva.property

// CHECK: sva.clocked_assume {{%.+}}, posedge {{%.+}} : !sva.property
sva.clocked_assume %p, posedge %clk : !sva.property

// CHECK: sva.clocked_cover {{%.+}}, posedge {{%.+}} : !sva.property
sva.clocked_cover %p, posedge %clk : !sva.property

//===----------------------------------------------------------------------===//
// Sequence Methods
//===----------------------------------------------------------------------===//

// CHECK: sva.seq.matched {{%.+}} : !sva.sequence -> i1
%matched = sva.seq.matched %s : !sva.sequence -> i1

// CHECK: sva.seq.triggered {{%.+}} : !sva.sequence -> i1
%triggered = sva.seq.triggered %s : !sva.sequence -> i1
