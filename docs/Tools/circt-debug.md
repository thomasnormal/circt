# circt-debug - Interactive Hardware Debugger

`circt-debug` is an interactive debugger for hardware designs, providing a GDB-like interface for stepping through simulations, setting breakpoints, and inspecting signal values.

## Overview

The CIRCT debugger provides:

- **Interactive CLI**: A command-line interface similar to GDB for hardware debugging
- **Breakpoints**: Support for line, signal, condition, cycle, and time breakpoints
- **Watchpoints**: Monitor signals for value changes
- **Scope Navigation**: Navigate the design hierarchy
- **Waveform Output**: Dump signals to VCD format
- **IDE Integration**: Debug Adapter Protocol (DAP) support for VS Code and other IDEs

## Quick Start

```bash
# Basic interactive debugging
circt-debug design.mlir --top myModule

# Start with VCD output
circt-debug design.mlir --top myModule --vcd trace.vcd

# Run commands from a file
circt-debug design.mlir -x commands.txt --batch

# Start in DAP mode for IDE integration
circt-debug design.mlir --dap
```

## Command Reference

### Execution Control

| Command | Alias | Description |
|---------|-------|-------------|
| `run [cycles]` | `r` | Run simulation for N cycles (0 = until end) |
| `step [n]` | `s` | Step N clock cycles (default: 1) |
| `stepi` | `si` | Step one delta cycle |
| `continue` | `c` | Continue until breakpoint or end |
| `reset` | | Reset simulation to initial state |

### Breakpoints

| Command | Alias | Description |
|---------|-------|-------------|
| `break <file:line>` | `b` | Set breakpoint at source location |
| `break -sig <signal>` | | Break on signal value change |
| `break -sig <signal> <value>` | | Break when signal equals value |
| `break -cond <expr>` | | Break when expression is true |
| `break -cycle <n>` | | Break at cycle N |
| `break -time <t>` | | Break at time T |

### Watchpoints

| Command | Alias | Description |
|---------|-------|-------------|
| `watch <signal>` | `w` | Watch signal for changes |

### Breakpoint/Watchpoint Management

| Command | Alias | Description |
|---------|-------|-------------|
| `delete [id]` | `d` | Delete breakpoint/watchpoint (all if no ID) |
| `enable <id>` | | Enable breakpoint/watchpoint |
| `disable <id>` | | Disable breakpoint/watchpoint |

### Inspection

| Command | Alias | Description |
|---------|-------|-------------|
| `print <expr>` | `p` | Print signal/expression value |
| `print/x <expr>` | | Print as hexadecimal |
| `print/b <expr>` | | Print as binary |
| `print/d <expr>` | | Print as decimal |
| `info signals` | `i sig` | List signals in current scope |
| `info breakpoints` | `i b` | List all breakpoints |
| `info watchpoints` | `i w` | List all watchpoints |
| `info scope` | | Show current scope |
| `info time` | | Show current simulation time |

### Scope Navigation

| Command | Description |
|---------|-------------|
| `scope` | Show current scope |
| `scope <path>` | Change to scope |
| `scope ..` | Go up one level |
| `list` | List child scopes and signals |

### Waveform Output

| Command | Alias | Description |
|---------|-------|-------------|
| `dump vcd <file>` | `x vcd` | Start VCD dump to file |
| `dump stop` | | Stop VCD dump |

### Other

| Command | Alias | Description |
|---------|-------|-------------|
| `set <signal> <value>` | | Force signal to value |
| `help [command]` | `h` | Show help |
| `quit` | `q` | Exit debugger |

## Breakpoint Examples

### Line Breakpoint

Set a breakpoint at a specific source location:

```
(circt-debug) break design.sv:42
Breakpoint 1 set: at design.sv:42
```

### Signal Breakpoint

Break when a signal changes:

```
(circt-debug) break -sig top.clk
Breakpoint 2 set: on signal top.clk

# Break when signal has specific value
(circt-debug) break -sig top.counter 8'hFF
Breakpoint 3 set: on signal top.counter == 0xff
```

### Conditional Breakpoint

Break when an expression is true:

```
(circt-debug) break -cond "counter > 100"
Breakpoint 4 set: when counter > 100
```

### Cycle Breakpoint

Break at a specific simulation cycle:

```
(circt-debug) break -cycle 1000
Breakpoint 5 set: at cycle 1000
```

## Value Formats

The debugger supports various value formats:

- **Decimal**: `123`, `456`
- **Hexadecimal**: `0xFF`, `'hABCD`
- **Binary**: `0b1010`, `'b1111`
- **Verilog-style**: `8'hFF`, `4'b1010`
- **With X/Z**: `4'b10x1`, `8'hxF`

## VCD Waveform Output

Start capturing waveforms:

```
(circt-debug) dump vcd trace.vcd
VCD dump started: trace.vcd

(circt-debug) run 100
... simulation runs ...

(circt-debug) dump stop
VCD dump stopped.
```

The VCD file can be viewed with GTKWave or other waveform viewers.

## IDE Integration (DAP)

The debugger implements the Debug Adapter Protocol for integration with VS Code and other IDEs.

### VS Code Configuration

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "circt-debug",
      "request": "launch",
      "name": "Debug Hardware",
      "program": "${workspaceFolder}/design.mlir",
      "topModule": "myModule",
      "args": []
    }
  ]
}
```

### DAP Features

- **Breakpoints**: Set in source files and on signals
- **Variable Inspection**: View signal values in the Variables pane
- **Hierarchy Navigation**: Browse design hierarchy in Modules view
- **Evaluate**: Evaluate expressions in Debug Console
- **Completions**: Auto-complete signal names

## Batch Mode

Run debugger commands from a file:

```bash
# Create a command file
cat > debug_commands.txt << EOF
break -cycle 100
run
print counter
continue
quit
EOF

# Run in batch mode
circt-debug design.mlir -x debug_commands.txt --batch
```

## Architecture

The debugger consists of several components:

1. **SimulationBackend**: Interface to simulation engines (Verilator, LLHD, etc.)
2. **DebugSession**: Manages the debug session, breakpoints, and state
3. **CommandProcessor**: Parses and executes CLI commands
4. **DAPServer**: Implements Debug Adapter Protocol for IDE integration
5. **VCDWriter**: Handles waveform output

### Signal Value Representation

The debugger uses 4-state logic values:
- `0`: Logic low
- `1`: Logic high
- `x`/`X`: Unknown
- `z`/`Z`: High impedance

### Simulation Time

Time is tracked with configurable precision:
- `fs`: Femtoseconds
- `ps`: Picoseconds
- `ns`: Nanoseconds (default)
- `us`: Microseconds
- `ms`: Milliseconds
- `s`: Seconds

## Extending the Debugger

### Adding a Simulation Backend

Implement the `SimulationBackend` interface:

```cpp
class MySimBackend : public SimulationBackend {
public:
  bool initialize(const DebugConfig &config) override;
  bool reset() override;
  bool stepDelta() override;
  bool stepClock() override;
  bool run(uint64_t cycles) override;
  bool runUntil(const SimTime &time) override;
  SimState &getState() override;
  const SimState &getState() const override;
  bool setSignal(StringRef path, const SignalValue &value) override;
  bool releaseSignal(StringRef path) override;
  bool isFinished() const override;
  StringRef getLastError() const override;
};
```

### Adding New Commands

Add a handler method in `CommandProcessor`:

```cpp
CommandResult CommandProcessor::cmdMyCommand(
    const std::vector<std::string> &args) {
  // Implementation
  return CommandResult::ok("Success\n");
}
```

Register it in `CommandProcessor::execute()`:

```cpp
if (cmd == "mycommand")
  return cmdMyCommand(args);
```

## Troubleshooting

### Common Issues

**Signal not found**
- Check the full hierarchical path
- Use `info signals` to list available signals
- Use `scope` to navigate the hierarchy

**Breakpoint not hit**
- Verify the breakpoint is enabled with `info breakpoints`
- Check that simulation reaches the condition

**VCD file empty**
- Ensure signals are changing
- Check that `dump stop` was called before exit

## See Also

- [circt-verilog](circt-verilog.md) - Verilog/SystemVerilog frontend
- [circt-lsp-server](circt-lsp-server.md) - Language Server Protocol support
- [VCD Format](https://en.wikipedia.org/wiki/Value_change_dump) - VCD specification
- [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/) - DAP specification
