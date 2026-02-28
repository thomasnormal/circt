# UVM Feature Coverage Matrix

Generated: 2026-02-28
Baseline: circt-sim commit cc7261bdb

## Summary

| Category | Verified | XFAIL | Total |
|----------|----------|-------|-------|
| Phase Lifecycle | 4 | 1 | 5 |
| Factory | 4 | 1 | 5 |
| Config DB | 4 | 2 | 6 |
| Sequences | 3 | 2 | 5 |
| TLM | 7 | 2 | 9 |
| Reporting | 5 | 0 | 5 |
| RAL | 2 | 3 | 5 |
| Objects | 4 | 1 | 5 |
| Components | 4 | 0 | 4 |
| Events | 2 | 0 | 2 |
| Coverage | 1 | 0 | 1 |
| Miscellaneous | 5 | 2 | 7 |
| **Total** | **45** | **14** | **59** |

## Detailed Feature Status

### Phase Lifecycle

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Phase lifecycle (9 phases) | PASS | crun/uvm-phase-all.sv | All 9 UVM phases execute in order |
| Phase objection raise/drop | PASS | crun/uvm-objection-nested.sv | Nested objections work correctly |
| Phase objection timeout | PASS | crun/uvm-phase-objection-timeout.sv | Drain time mechanism works |
| Phase domain (common) | PASS | crun/uvm-phase-domain-common.sv | get_common_domain() returns valid handle |
| Phase jump forward | XFAIL | crun/uvm-phase-jump-forward.sv | phase.jump() hangs — not implemented |

### Factory

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Factory create | PASS | crun/uvm-factory-create.sv | type_id::create() produces correct type |
| Factory override (type) | PASS | uvm/uvm_factory_override_test.sv | set_type_override works |
| Factory override chain | PASS | crun/uvm-factory-override-chain.sv | Override chaining A→B→C works |
| Factory override inst path | PASS | crun/uvm-factory-override-inst-path.sv | Instance path override works |
| Factory set_type_override_by_type | XFAIL | crun/uvm-factory-type-override.sv | Doesn't produce derived class |

### Config DB

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Config DB set/get (int) | PASS | uvm/config_db_test.sv | Integer values work |
| Config DB set/get (string) | PASS | crun/uvm-config-db.sv | String values work |
| Config DB hierarchical | PASS | crun/uvm-config-db-hierarchical.sv | Hierarchical set/get works |
| Config DB with uvm_object | PASS | crun/uvm-config-db-object.sv | Object type storage works |
| Config DB virtual interface | XFAIL | crun/uvm-config-db-virtual-if.sv | Virtual interface not supported in config_db |
| Config DB wildcard matching | XFAIL | crun/uvm-config-db-wildcard.sv | uvm_is_match wildcard broken |

### Sequences

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Sequence start/body | PASS | uvm/uvm_sequence_test.sv | Basic sequence execution works |
| Sequence get_response | PASS | crun/uvm-sequence-response.sv | get_response() with set_id_info() works |
| Sequencer arbitration | PASS | crun/uvm-sequencer-arbitration.sv | SEQ_ARB_FIFO arbitration works |
| Virtual sequencer | XFAIL | crun/uvm-sequence-virtual.sv | Virtual sequencer pattern not supported |
| Sequence library | XFAIL | crun/uvm-sequence-library.sv | uvm_sequence_library not in bundled UVM |

### TLM

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| TLM port connect | PASS | uvm/uvm_tlm_port_test.sv | Basic TLM port connection works |
| TLM FIFO | PASS | crun/uvm-tlm-fifo.sv | Basic TLM FIFO works |
| TLM FIFO (bounded) | PASS | crun/uvm-tlm-fifo-bounded.sv | Bounded FIFO capacity works (was "known broken") |
| TLM FIFO test | PASS | uvm/uvm_tlm_fifo_test.sv | Extended TLM FIFO testing |
| Analysis port broadcast | PASS | crun/uvm-tlm-analysis-broadcast.sv | analysis_port to 3 subscribers works |
| Analysis port basic | PASS | crun/uvm-analysis-port.sv | Analysis port write/receive works |
| TLM req/rsp channel | PASS | crun/uvm-tlm-req-rsp-channel.sv | TLM req/rsp channel works |
| Blocking transport | PASS | crun/uvm-tlm-transport.sv | Blocking transport port works |
| Analysis FIFO | XFAIL | crun/uvm-analysis-fifo.sv | analysis_fifo write path broken |
| TLM-2.0 blocking | XFAIL | crun/uvm-tlm2-blocking.sv | TLM-2.0 classes not in bundled UVM |

### Reporting

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| UVM reporting | PASS | uvm/uvm_reporting_test.sv | uvm_info/warning/error work |
| Report severity count | PASS | crun/uvm-report-severity-count.sv | Severity count tracking works |
| Report max quit count | PASS | crun/uvm-report-max-quit.sv | Max quit count termination works |
| Report catcher demotion | PASS | crun/uvm-report-catcher-demote.sv | Report catcher severity demotion works |
| String to severity | PASS | uvm/uvm_string_to_severity_test.sv | Severity string conversion works |

### RAL (Register Abstraction Layer)

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| RAL basic | PASS | crun/uvm-ral-basic.sv | Basic register model setup works |
| RAL block/map/register | PASS | crun/uvm-ral-block-map.sv | Block/map/register address setup works |
| RAL adapter | XFAIL | crun/uvm-ral-adapter.sv | RAL adapter subsystem not fully supported |
| RAL mirrored values | XFAIL | crun/uvm-ral-mirrored.sv | Mirrored value tracking not implemented |
| RAL reg read/write | XFAIL | crun/uvm-ral-reg-read-write.sv | Frontdoor access not supported |

### Objects

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Object utils macros | PASS | crun/uvm-object-utils.sv | `uvm_object_utils works |
| Object field automation | PASS | crun/uvm-object-field-automation.sv | Object copy/compare works |
| Transaction recording | PASS | crun/uvm-transaction-record.sv | begin_tr/end_tr recording works |
| Printer | PASS | crun/uvm-printer.sv | Object printing works |
| Object pack/unpack | XFAIL | crun/uvm-object-pack-unpack.sv | Bit ordering issues |

### Components

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Component hierarchy | PASS | crun/uvm-component-hierarchy.sv | Parent/child hierarchy works |
| Component find_all | PASS | crun/uvm-component-find-all.sv | find_all() component search works |
| Component get_type_name | PASS | crun/uvm-component-type-name.sv | get_type_name() works |
| Component suspend/resume | PASS | uvm/uvm_component_suspend_resume_test.sv | Component suspend/resume works |

### Events

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| UVM event trigger/wait | PASS | crun/uvm-event.sv | Event trigger and wait work |
| Event pool | PASS | crun/uvm-event-pool.sv | Event pool get/trigger works |

### Coverage

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Coverage collection | PASS | uvm/uvm_coverage_test.sv | Basic coverage works |

### Miscellaneous

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Simple UVM test | PASS | uvm/uvm_simple_test.sv | Basic UVM test lifecycle |
| Basic UVM (crun) | PASS | crun/uvm-basic.sv | Basic crun UVM test |
| Callback mechanism | PASS | uvm/uvm_callback_test.sv | UVM callbacks work |
| Comparator | PASS | uvm/uvm_comparator_test.sv | in-order comparator works |
| Resource DB | PASS | crun/uvm-resource-db.sv | uvm_resource_db set/get works |
| DAP (data access policy) | XFAIL | crun/uvm-dap-basic.sv | DAP classes not in bundled UVM |
| Push driver | XFAIL | crun/uvm-push-driver.sv | uvm_push_driver not in bundled UVM |

### Additional Existing Tests (test/Runtime/uvm/)

| Feature | Status | Test File | Notes |
|---------|--------|-----------|-------|
| Factory basic | PASS | uvm/uvm_factory_test.sv | Factory creation and lookup |
| Sequencer | PASS | uvm/uvm_sequencer_test.sv | Sequencer start_item/finish_item |
| Send request | PASS | uvm/uvm_send_request_test.sv | Sequence send_request path |
| Port connect semantic | PASS | uvm/uvm_port_connect_semantic_test.sv | Port connection semantics |
| Phase aliases | PASS | uvm/uvm_phase_aliases_test.sv | Phase alias resolution |
| Phase add scope | PASS | uvm/uvm_phase_add_scope_validation_test.sv | Phase scope validation |
| Phase set_jump null active | PASS | uvm/uvm_phase_set_jump_null_active_test.sv | set_jump(null) handling |
| Phase wait_for_state | PASS | uvm/uvm_phase_wait_for_state_test.sv | Phase wait_for_state works |
| Run phase time-zero guard | PASS | uvm/uvm_run_phase_time_zero_guard_test.sv | Time-zero guard works |
| Timeout plusarg | PASS | uvm/uvm_timeout_plusarg_test.sv | +UVM_TIMEOUT handling |
| Stress test | PASS | uvm/uvm_stress_test.sv | Multi-component stress test |
| RAL test | PASS | uvm/uvm_ral_test.sv | RAL basic integration |
| Pool | PASS | crun/uvm-pool.sv | uvm_pool parameterized container |
| Heartbeat | PASS | crun/uvm-heartbeat.sv | UVM heartbeat mechanism |
| Path flag | PASS | crun/uvm-path-flag.sv | UVM path flag handling |

## Known Limitations

1. **Virtual interface in config_db** — config_db cannot store/retrieve virtual interface handles; virtual interface types are not fully supported in the MLIR lowering.

2. **Config DB wildcard matching** — `uvm_is_match()` glob/regex matching is broken; only exact string matches work in config_db paths.

3. **Phase jumping** — `phase.jump()` hangs the simulator; phase state machine doesn't support non-sequential transitions.

4. **RAL frontdoor/mirrored values** — Register abstraction layer supports basic model setup and address maps, but frontdoor read/write operations and mirrored value tracking are not implemented.

5. **TLM-2.0** — TLM-2.0 classes (`uvm_tlm_b_initiator_socket`, etc.) are not included in the bundled UVM library.

6. **DAP classes** — `uvm_set_before_get_dap` and related data access policy classes are not in the bundled UVM library.

7. **Push driver** — `uvm_push_driver` and `uvm_push_sequencer` are not in the bundled UVM library.

8. **Sequence library** — `uvm_sequence_library` is not in the bundled UVM library.

9. **Object pack/unpack** — Bit ordering in `pack()`/`unpack()` operations does not match the IEEE specification.

10. **Factory set_type_override_by_type** — `set_type_override_by_type()` correctly registers the override but `create()` still returns the base type.

11. **Virtual sequencer** — Virtual sequencer pattern (`p_sequencer` cast and sub-sequencer access) is not supported.

12. **Analysis FIFO** — `uvm_tlm_analysis_fifo` write path is broken; items written via analysis port are not received.

## Prioritized Fix List

1. **Config DB wildcard matching** — Affects testbench portability; many UVM environments use wildcard paths.
2. **Factory set_type_override_by_type** — Core UVM pattern; blocks factory-based testbench reuse.
3. **Analysis FIFO** — Common verification pattern; blocks scoreboards using analysis FIFOs.
4. **Virtual sequencer** — Required for multi-interface testbenches.
5. **Object pack/unpack** — Needed for protocol-level sequence items.
6. **Phase jumping** — Used in some advanced test topologies for early termination.
7. **RAL frontdoor access** — Required for register-based verification.
8. **RAL mirrored values** — Required for register checking.
9. **Config DB virtual interface** — Common pattern but has workarounds.
10. **Bundled UVM library gaps** (TLM-2.0, DAP, push driver, sequence library) — Add missing classes to bundled UVM.
