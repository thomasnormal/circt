# UVM Feature Coverage Matrix

Generated: 2026-02-28
Baseline: circt-sim commit cc7261bdb

## Overview

This matrix tracks UVM feature coverage across two test suites:

- **test/Tools/crun/uvm-*.sv** — 168 targeted feature tests (147 PASS, 21 XFAIL)
- **test/Runtime/uvm/\*.sv** — 26 integration/regression tests (26 PASS, 0 XFAIL)

**Grand total: 194 tests, 173 PASS, 21 XFAIL**

## Summary by Category

| Category | crun PASS | crun XFAIL | crun Total | Runtime PASS | Grand Total |
|----------|-----------|------------|------------|--------------|-------------|
| Phase Lifecycle | 12 | 1 | 13 | 8 | 21 |
| Objection | 4 | 0 | 4 | 1 | 5 |
| Factory | 8 | 2 | 10 | 2 | 12 |
| Config DB | 9 | 2 | 11 | 1 | 12 |
| Sequences | 11 | 4 | 15 | 4 | 19 |
| TLM / Analysis | 22 | 0 | 22 | 2 | 24 |
| RAL | 18 | 1 | 19 | 1 | 20 |
| Reporting | 2 | 2 | 4 | 2 | 6 |
| Objects | 10 | 1 | 11 | 0 | 11 |
| Field Macros | 5 | 0 | 5 | 0 | 5 |
| Components | 6 | 2 | 8 | 1 | 9 |
| Events | 3 | 0 | 3 | 0 | 3 |
| Barriers / Pools / Queues | 8 | 0 | 8 | 0 | 8 |
| Integration | 8 | 4 | 12 | 3 | 15 |
| Miscellaneous | 21 | 2 | 23 | 1 | 24 |
| **Total** | **147** | **21** | **168** | **26** | **194** |

---

## Detailed Feature Status

### Phase Lifecycle (13 crun + 8 Runtime = 21 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-phase-all.sv | PASS | All 9 UVM phases execute in order |
| crun/uvm-phase-objection-timeout.sv | PASS | Drain time mechanism works |
| crun/uvm-phase-domain-common.sv | PASS | get_common_domain() returns valid handle |
| crun/uvm-phase-domain-list.sv | PASS | Domain list enumeration works |
| crun/uvm-phase-double-raise.sv | PASS | Double raise handled correctly |
| crun/uvm-phase-drop-without-raise.sv | PASS | Drop without raise (no crash) |
| crun/uvm-phase-extract-report.sv | PASS | extract and report phases run in order |
| crun/uvm-phase-custom-function.sv | PASS | Custom function phase executes |
| crun/uvm-phase-custom-topdown.sv | PASS | Custom top-down phase executes |
| crun/uvm-phase-raise-in-multiple.sv | PASS | Raise from multiple components |
| crun/uvm-phase-zero-time.sv | PASS | Phase completing at time zero |
| crun/uvm-phase-objection-after-drop.sv | XFAIL | Class method references module-scope clk — slang "unknown name `clk`" |
| crun/uvm-phase-jump-forward.sv | PASS | phase.jump() to extract_phase (previously XFAIL, now fixed) |
| uvm/uvm_phase_aliases_test.sv | PASS | Phase alias resolution |
| uvm/uvm_phase_add_scope_validation_test.sv | PASS | Phase scope validation |
| uvm/uvm_phase_set_jump_null_active_test.sv | PASS | set_jump(null) on null active phase handled |
| uvm/uvm_phase_wait_for_state_test.sv | PASS | wait_for_state() works |
| uvm/uvm_phase_ordering_semantic_test.sv | PASS | Phase ordering semantics |
| uvm/uvm_run_phase_time_zero_guard_test.sv | PASS | Time-zero guard for run phase |
| uvm/uvm_timeout_plusarg_test.sv | PASS | +UVM_TIMEOUT plusarg handling |
| uvm/uvm_simple_test.sv | PASS | Basic UVM test lifecycle |

### Objection (4 crun + 1 Runtime = 5 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-objection-nested.sv | PASS | Nested objections work correctly |
| crun/uvm-objection-callback.sv | PASS | Objection raised/dropped callbacks fire |
| crun/uvm-objection-callback-drain.sv | PASS | Drain callback fires after final drop |
| crun/uvm-objection-count.sv | PASS | get_objection_count() tracking works |
| uvm/uvm_objection_count_semantic_test.sv | PASS | Per-component and total objection count semantics |

### Factory (10 crun + 2 Runtime = 12 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-factory-create.sv | PASS | type_id::create() produces correct type |
| crun/uvm-factory-override-chain.sv | PASS | Override chaining A→B→C works |
| crun/uvm-factory-double-override.sv | PASS | Double override: last-wins semantics |
| crun/uvm-factory-override-priority.sv | PASS | Override priority ordering |
| crun/uvm-factory-parameterized.sv | PASS | Parameterized type factory create |
| crun/uvm-factory-type-override.sv | PASS | set_type_override_by_type() (previously XFAIL, now fixed) |
| crun/uvm-factory-create-null-parent.sv | PASS | create() with null parent |
| crun/uvm-factory-create-unknown.sv | PASS | create() unknown type returns null |
| crun/uvm-factory-create-by-name.sv | XFAIL | uvm_coreservice_t::get().get_factory() not available — undeclared identifier |
| crun/uvm-factory-override-inst-path.sv | XFAIL | Instance path override not supported in circt-sim |
| uvm/uvm_factory_test.sv | PASS | Factory creation and lookup |
| uvm/uvm_factory_override_test.sv | PASS | set_type_override works |

### Config DB (11 crun + 1 Runtime = 12 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-config-db.sv | PASS | String values work |
| crun/uvm-config-db-hierarchical.sv | PASS | Hierarchical set/get works |
| crun/uvm-config-db-object.sv | PASS | Object type storage works |
| crun/uvm-config-db-wildcard.sv | PASS | Wildcard "*" path matching (previously XFAIL, now fixed) |
| crun/uvm-config-db-multiple-keys.sv | PASS | Multiple keys in same component |
| crun/uvm-config-db-precedence.sv | PASS | Precedence: more specific path wins |
| crun/uvm-config-db-empty-key.sv | PASS | Empty key string handled |
| crun/uvm-config-db-get-before-set.sv | PASS | get() before set() returns false |
| crun/uvm-config-db-null-context.sv | PASS | Null context (global scope) set/get |
| crun/uvm-config-db-type-mismatch.sv | XFAIL | Type checking not enforced across parameterized specializations |
| crun/uvm-config-db-virtual-if.sv | XFAIL | Virtual interface type: slang "unsupported arbitrary symbol reference" |
| uvm/config_db_test.sv | PASS | Integer/string/object types work |

### Sequences (15 crun + 4 Runtime = 19 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-sequence-response.sv | PASS | get_response() with set_id_info() works |
| crun/uvm-sequencer-arbitration.sv | PASS | SEQ_ARB_FIFO arbitration works |
| crun/uvm-sequence-create-send.sv | PASS | create_item + send_request path |
| crun/uvm-sequence-do-with.sv | PASS | `uvm_do_with randomization constraint |
| crun/uvm-sequence-body-return.sv | PASS | body() return value |
| crun/uvm-sequence-pre-post-body.sv | PASS | pre_body/post_body hooks |
| crun/uvm-sequence-item-clone.sv | XFAIL | clone() requires `uvm_field_int automation macros not used |
| crun/uvm-sequence-item-no-sequencer.sv | PASS | Item creation without sequencer |
| crun/uvm-sequence-nested-3deep.sv | PASS | 3-level nested sequence execution |
| crun/uvm-sequence-lock-grab.sv | PASS | lock()/grab() sequencer access |
| crun/uvm-sequence-priority.sv | PASS | Priority-based arbitration |
| crun/uvm-sequence-library-manual.sv | PASS | Manual sequence library (no uvm_sequence_library) |
| crun/uvm-sequence-virtual.sv | XFAIL | Virtual sequencer pattern (p_sequencer cast) not supported |
| crun/uvm-sequence-library.sv | XFAIL | uvm_sequence_library not in bundled UVM |
| crun/uvm-sequence-no-driver.sv | XFAIL | Class method references module-scope clk — slang "unknown name `clk`" |
| uvm/uvm_sequence_test.sv | PASS | Basic sequence execution |
| uvm/uvm_sequencer_test.sv | PASS | Sequencer start_item/finish_item |
| uvm/uvm_send_request_test.sv | PASS | Sequence send_request path |
| uvm/uvm_port_connect_semantic_test.sv | PASS | Port connection semantics |

### TLM / Analysis (22 crun + 2 Runtime = 24 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-tlm-fifo.sv | PASS | Basic TLM FIFO works |
| crun/uvm-tlm-fifo-bounded.sv | PASS | Bounded FIFO capacity works |
| crun/uvm-tlm-fifo-flush.sv | PASS | Flush clears FIFO |
| crun/uvm-tlm-fifo-get-empty.sv | PASS | get() blocks on empty FIFO |
| crun/uvm-tlm-fifo-interleave.sv | PASS | Interleaved put/get works |
| crun/uvm-tlm-fifo-peek.sv | PASS | peek() returns item without removing |
| crun/uvm-tlm-fifo-size-query.sv | PASS | size()/used()/is_full() queries |
| crun/uvm-tlm-analysis-broadcast.sv | PASS | analysis_port to 3 subscribers |
| crun/uvm-tlm-analysis-100.sv | PASS | Analysis port with 100 subscribers |
| crun/uvm-tlm-analysis-multi.sv | PASS | Multiple analysis ports in env |
| crun/uvm-analysis-port.sv | PASS | Analysis port write/receive |
| crun/uvm-analysis-fifo.sv | PASS | analysis_fifo write path (previously XFAIL, now fixed) |
| crun/uvm-analysis-imp-decl.sv | PASS | `uvm_analysis_imp_decl macro |
| crun/uvm-tlm-req-rsp-channel.sv | PASS | TLM req/rsp channel |
| crun/uvm-tlm-transport.sv | PASS | Blocking transport port |
| crun/uvm-tlm-transport-channel.sv | PASS | Transport channel |
| crun/uvm-tlm-nonblocking.sv | PASS | Non-blocking TLM port try_put/try_get |
| crun/uvm-tlm-port-not-connected.sv | PASS | Unconnected port warning (no crash) |
| crun/uvm-tlm-put-get-ports.sv | PASS | Put/get port pairing |
| crun/uvm-tlm2-payload-only.sv | PASS | TLM-2.0 generic payload fields |
| crun/uvm-tlm2-time-only.sv | PASS | TLM-2.0 time annotation |
| crun/uvm-tlm2-blocking.sv | PASS | TLM-2.0 b_transport socket works |
| uvm/uvm_tlm_fifo_test.sv | PASS | Extended TLM FIFO testing |
| uvm/uvm_tlm_port_test.sv | PASS | Basic TLM port connection |

### RAL — Register Abstraction Layer (19 crun + 1 Runtime = 20 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-ral-basic.sv | PASS | Basic register model setup |
| crun/uvm-ral-block-map.sv | PASS | Block/map/register address setup |
| crun/uvm-ral-block-find.sv | PASS | get_reg_by_name() find in block |
| crun/uvm-ral-field-access.sv | PASS | Field access modes (RW, RO, WO) |
| crun/uvm-ral-map-submaps.sv | PASS | Sub-map hierarchy |
| crun/uvm-ral-mem-basic.sv | PASS | uvm_mem basic (size, access) |
| crun/uvm-ral-mem-configure.sv | PASS | uvm_mem configure() |
| crun/uvm-ral-multi-reg-block.sv | PASS | Multiple register blocks |
| crun/uvm-ral-predictor-basic.sv | PASS | Predictor instantiation |
| crun/uvm-ral-predictor-setup.sv | PASS | Predictor connect() setup |
| crun/uvm-ral-reg-callbacks.sv | PASS | uvm_reg_cbs pre/post callbacks |
| crun/uvm-ral-reg-coverage.sv | PASS | Register coverage sampling |
| crun/uvm-ral-reg-field-configure.sv | PASS | Field configure() with access/reset |
| crun/uvm-ral-reg-reset.sv | PASS | Register reset() and get_reset() |
| crun/uvm-ral-submap-create.sv | PASS | Sub-map creation and address |
| crun/uvm-ral-mirrored.sv | PASS | Mirrored value predict() (previously XFAIL, now fixed) |
| crun/uvm-ral-reg-read-write.sv | PASS | reg.write()/reg.read() via stub adapter (previously XFAIL, now fixed) |
| crun/uvm-ral-adapter.sv | PASS | RAL adapter subclass and register block creation works |
| crun/uvm-ral-reg-callback-add.sv | XFAIL | uvm_reg nested class triggers slang non-static member access error |
| uvm/uvm_ral_test.sv | PASS | RAL basic integration |

### Reporting (4 crun + 2 Runtime = 6 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-report-severity-count.sv | PASS | Severity count tracking |
| crun/uvm-report-catcher-demote.sv | PASS | Report catcher severity demotion |
| crun/uvm-report-fatal-catch.sv | XFAIL | uvm_report_catcher pure virtual 'catch' — slang rejects |
| crun/uvm-report-max-quit.sv | XFAIL | set_max_quit_count() not fully implemented in report server |
| uvm/uvm_reporting_test.sv | PASS | uvm_info/warning/error work |
| uvm/uvm_string_to_severity_test.sv | PASS | Severity string conversion |

### Objects (11 crun = 11 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-object-utils.sv | PASS | `uvm_object_utils macro |
| crun/uvm-object-field-automation.sv | PASS | Object copy/compare via field macros |
| crun/uvm-object-clone.sv | PASS | clone() deep-copies object |
| crun/uvm-object-compare-deep.sv | PASS | compare() deep field comparison |
| crun/uvm-object-compare-null.sv | PASS | compare() with null reference |
| crun/uvm-object-convert2string.sv | PASS | convert2string() method |
| crun/uvm-object-copy-independence.sv | PASS | copy() produces independent copy |
| crun/uvm-object-copy-null.sv | PASS | copy(null) handled gracefully |
| crun/uvm-object-do-methods.sv | PASS | do_copy/do_compare/do_print overrides |
| crun/uvm-object-print.sv | PASS | Object print to string |
| crun/uvm-object-pack-unpack.sv | XFAIL | `uvm_field_int pack/unpack — bit ordering issues |

### Field Macros (5 crun = 5 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-field-int-only.sv | PASS | `uvm_field_int automation |
| crun/uvm-field-string-only.sv | PASS | `uvm_field_string automation |
| crun/uvm-field-enum.sv | PASS | `uvm_field_enum automation |
| crun/uvm-field-array.sv | PASS | `uvm_field_array_int automation |
| crun/uvm-field-string-object.sv | PASS | Mixed string+object field macros |

### Components (8 crun + 1 Runtime = 9 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-component-hierarchy.sv | PASS | Parent/child hierarchy |
| crun/uvm-component-find-all.sv | PASS | find_all() component search |
| crun/uvm-component-type-name.sv | PASS | get_type_name() |
| crun/uvm-component-children.sv | PASS | get_child()/get_num_children() |
| crun/uvm-component-duplicate-name.sv | PASS | Duplicate child name warning |
| crun/uvm-component-get-child-nonexist.sv | PASS | get_child() non-existent returns null |
| crun/uvm-component-deep-hierarchy.sv | XFAIL | get_full_name() returns ".uvm_test_top" (leading dot) instead of "uvm_test_top" |
| crun/uvm-component-lookup-regex.sv | XFAIL | find_all() not available in our UVM library version |
| uvm/uvm_component_suspend_resume_test.sv | PASS | Component suspend/resume |

### Events (3 crun = 3 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-event.sv | PASS | Event trigger and wait |
| crun/uvm-event-pool.sv | PASS | Event pool get/trigger |
| crun/uvm-event-data.sv | PASS | Event trigger with data payload |

### Barriers / Pools / Queues (8 crun = 8 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-barrier.sv | PASS | uvm_barrier basic wait |
| crun/uvm-barrier-no-wait.sv | PASS | Barrier with all processes present at threshold |
| crun/uvm-barrier-pool.sv | PASS | uvm_barrier_pool get/wait |
| crun/uvm-barrier-threshold-1.sv | PASS | Barrier with threshold=1 |
| crun/uvm-pool.sv | PASS | uvm_pool parameterized container |
| crun/uvm-pool-operations.sv | PASS | Pool add/exists/delete/num |
| crun/uvm-queue.sv | PASS | uvm_queue push/pop |
| crun/uvm-queue-iterator.sv | PASS | uvm_queue iteration |

### Integration (12 crun + 3 Runtime = 15 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-integ-callbacks-factory.sv | PASS | Callbacks combined with factory override |
| crun/uvm-integ-config-phase-report.sv | PASS | Config DB + phase + reporting pipeline |
| crun/uvm-integ-env-config-factory.sv | PASS | Env + config + factory combined |
| crun/uvm-integ-event-sequence-sync.sv | PASS | Event synchronization across sequences |
| crun/uvm-integ-objection-sequence.sv | PASS | Sequence-driven objection management |
| crun/uvm-integ-report-phase-lifecycle.sv | PASS | Reporting across full phase lifecycle |
| crun/uvm-integ-stress-create.sv | PASS | Stress: many component creates in build |
| crun/uvm-integ-tlm-pipeline.sv | PASS | Full TLM pipeline (seq→drv→sb via analysis) |
| crun/uvm-integ-factory-config-seq.sv | XFAIL | Class method references module-scope clk — slang "unknown name `clk`" |
| crun/uvm-integ-multi-agent.sv | XFAIL | Class method references module-scope clk — slang "unknown name `clk`" |
| crun/uvm-integ-ral-config-env.sv | XFAIL | uvm_reg nested class triggers slang non-static member access error |
| crun/uvm-integ-seq-driver-scoreboard.sv | XFAIL | Class method references module-scope clk — slang "unknown name `clk`" |
| uvm/uvm_stress_test.sv | PASS | Multi-component stress test |
| uvm/uvm_callback_test.sv | PASS | UVM callbacks end-to-end |
| uvm/uvm_comparator_test.sv | PASS | In-order comparator |

### Miscellaneous (23 crun + 1 Runtime = 24 total)

| Test File | Status | Notes |
|-----------|--------|-------|
| crun/uvm-basic.sv | PASS | Basic UVM test with driver/monitor |
| crun/uvm-algorithmic-comparator.sv | XFAIL | uvm_algorithmic_comparator parameterized type instantiation errors |
| crun/uvm-comparer.sv | PASS | uvm_comparer configuration |
| crun/uvm-coreservice.sv | PASS | uvm_coreservice_t access |
| crun/uvm-driver-basic.sv | PASS | Driver start_item/finish_item |
| crun/uvm-env-agent.sv | PASS | Env containing agent subcomponent |
| crun/uvm-globals-wait-nba.sv | PASS | uvm_wait_for_nba_region() |
| crun/uvm-heartbeat.sv | PASS | UVM heartbeat mechanism |
| crun/uvm-monitor-basic.sv | PASS | Monitor collect/write to analysis |
| crun/uvm-pair.sv | PASS | uvm_class_pair and uvm_built_in_pair |
| crun/uvm-path-flag.sv | PASS | UVM_HIER/UVM_FULL path flag |
| crun/uvm-printer.sv | PASS | Object printing |
| crun/uvm-recorder.sv | PASS | Transaction recording |
| crun/uvm-resource-db.sv | PASS | uvm_resource_db set/get |
| crun/uvm-root-find.sv | PASS | uvm_root::find() |
| crun/uvm-scoreboard-basic.sv | PASS | Scoreboard with analysis imp |
| crun/uvm-spell-chkr.sv | PASS | uvm_spell_chkr string matching |
| crun/uvm-transaction-record.sv | PASS | begin_tr/end_tr recording |
| crun/uvm-transaction-timing.sv | PASS | Transaction accept/request/response timing |
| crun/uvm-tree-printer.sv | PASS | uvm_tree_printer output |
| crun/uvm-dap-basic.sv | PASS | set_before_get_dap and get_to_lock_dap work |
| crun/uvm-pack-manual.sv | XFAIL | uvm_packer::get_bits/put_bits not in bundled UVM |
| crun/uvm-push-driver.sv | PASS | uvm_push_driver receives items via put interface |
| uvm/uvm_coverage_test.sv | PASS | Basic coverage collection |

---

## Known Limitations

The 21 XFAIL tests fall into four root-cause categories:

### 1. Bundled UVM Library Gaps
Classes not present in the bundled UVM library version.

- **Sequence library** (`uvm-sequence-library.sv`) — `uvm_sequence_library` not included.
- **uvm_packer low-level API** (`uvm-pack-manual.sv`) — `get_bits()`/`put_bits()` not available.
- **component find_all regex** (`uvm-component-lookup-regex.sv`) — `find_all()` with regex not available in library version.

**Fix**: Upgrade or augment the bundled UVM library.

### 2. Slang Compilation Errors
Tests that cannot compile due to limitations in the slang→MLIR lowering.

- **Module-scope clock reference in class** (`uvm-phase-objection-after-drop.sv`,
  `uvm-sequence-no-driver.sv`, `uvm-integ-factory-config-seq.sv`,
  `uvm-integ-multi-agent.sv`, `uvm-integ-seq-driver-scoreboard.sv`) — slang reports
  "unknown name `clk`" when a class method references a module-scope interface or signal.
- **uvm_reg nested class** (`uvm-ral-reg-callback-add.sv`,
  `uvm-integ-ral-config-env.sv`) — slang emits non-static member access error for
  nested classes inside `uvm_reg`.
- **Virtual interface in config_db** (`uvm-config-db-virtual-if.sv`) — slang reports
  "unsupported arbitrary symbol reference" for module-scope interface instance in class context.
- **Report catcher override** (`uvm-report-fatal-catch.sv`) — slang rejects override of
  pure virtual method `catch` in `uvm_report_catcher` subclass.
- **uvm_algorithmic_comparator parameterized types** (`uvm-algorithmic-comparator.sv`) —
  parameterized type instantiation errors with `uvm_algorithmic_comparator #(BEFORE, TRANSFORMER, AFTER)`.
- **Sequence item clone()** (`uvm-sequence-item-clone.sv`) — clone() requires `uvm_field_int`
  field automation macros to be declared; manual copy approach not supported by clone interceptor.

**Fix**: Improve slang/ImportVerilog lowering for these SV patterns.

### 3. circt-sim Feature Gaps
Features supported by UVM but not yet implemented in circt-sim.

- **Factory instance path override** (`uvm-factory-override-inst-path.sv`) —
  `set_inst_override_by_type()` with hierarchical path not supported.
- **Factory create by name** (`uvm-factory-create-by-name.sv`) —
  `uvm_coreservice_t::get().get_factory()` access pattern not available.
- **Virtual sequencer** (`uvm-sequence-virtual.sv`) — `p_sequencer` cast to
  virtual sequencer subtype not supported.
- **set_max_quit_count()** (`uvm-report-max-quit.sv`) — max quit termination
  logic not implemented in report server.
- **get_full_name() leading dot** (`uvm-component-deep-hierarchy.sv`) —
  returns ".uvm_test_top" instead of "uvm_test_top" at top level.

**Fix**: Implement missing features in circt-sim / LLHDProcessInterpreter.

### 4. Object Pack/Unpack
- **Bit ordering** (`uvm-object-pack-unpack.sv`) — `pack()`/`unpack()` bit ordering
  does not match the IEEE 1800 specification (uses `uvm_field_int` automation).
- **Config DB type checking** (`uvm-config-db-type-mismatch.sv`) — the config_db
  interceptor does not enforce type checking across different parameterized specializations.

---

## Prioritized Fix List

Items ordered by expected impact and implementation difficulty:

1. **Module-scope clock reference in class** (5 tests) — Fix slang lowering to handle
   class methods that reference module-scope interface signals. Unblocks 5 integration tests.

2. **uvm_reg nested class** (2 tests) — Fix slang lowering for nested classes inside
   `uvm_reg`. Unblocks RAL reg callback-add and RAL integration tests.

3. **get_full_name() leading dot** (1 test) — Off-by-one in component path construction.
   Easy fix in LLHDProcessInterpreter.

4. **Factory instance path override** (1 test) — Implement `set_inst_override_by_type()`
   path matching in the factory interceptor.

5. **Factory create by name** (1 test) — Expose `get_factory()` via coreservice interceptor.

6. **set_max_quit_count()** (1 test) — Implement quit count tracking in report server.

7. **Virtual sequencer p_sequencer cast** (1 test) — Support downcasting `m_sequencer`
   to a subtype via `p_sequencer`.

8. **Object pack/unpack bit ordering** (1 test) — Fix `uvm_field_int` pack/unpack bit ordering.

9. **Config DB type mismatch enforcement** (1 test) — Enforce type tags in config_db interceptor.

10. **Bundled UVM library gaps** (3 tests) — Add missing classes to the bundled UVM:
    sequence library, packer low-level API, find_all() regex.

11. **Virtual interface in config_db** (1 test) — Requires full virtual interface type support
    in the MLIR lowering and config_db interceptor.

12. **Report catcher pure virtual** (1 test) — Fix slang handling of pure virtual override
    in uvm_report_catcher subclass.
