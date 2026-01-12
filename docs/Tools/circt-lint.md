# circt-lint

The `circt-lint` tool provides configurable linting for Verilog and SystemVerilog designs. It can be used standalone or integrated into CI/CD pipelines.

## Overview

`circt-lint` analyzes Verilog/SystemVerilog source files and reports potential issues, coding style violations, and common mistakes. It supports multiple output formats and integrates with project configuration files.

## Features

- **Configurable Rules**: Enable/disable rules and set severity levels
- **Multiple Output Formats**: Text, JSON, SARIF, JUnit XML
- **Project Configuration**: Automatic discovery of `circt-project.yaml`
- **CI/CD Integration**: JUnit XML and SARIF for GitHub Actions, Jenkins, etc.
- **Incremental Analysis**: Only re-analyze changed files

## Usage

### Basic Usage

```bash
# Lint all files in current directory
circt-lint .

# Lint specific files
circt-lint rtl/module.sv rtl/top.sv

# Use a specific configuration
circt-lint --config=lint.yaml rtl/
```

### Output Formats

```bash
# Text output (default)
circt-lint .

# JSON output
circt-lint --format=json .

# SARIF output (for GitHub Code Scanning)
circt-lint --format=sarif --output=results.sarif .

# JUnit XML output (for CI/CD)
circt-lint --junit-xml=results.xml .
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--config=<file>` | Path to lint configuration file |
| `--format=<fmt>` | Output format: text, json, sarif |
| `--output=<file>` | Output file (default: stdout) |
| `--junit-xml=<file>` | Generate JUnit XML report |
| `--target=<name>` | Use target-specific configuration |
| `--rule=<name>` | Enable a specific rule |
| `--no-rule=<name>` | Disable a specific rule |
| `--severity=<rule>:<level>` | Set rule severity |
| `-W<rule>` | Enable rule as warning |
| `-E<rule>` | Enable rule as error |
| `--verbose` | Verbose output |

## Configuration

### Lint Configuration File

Create `lint.yaml` or `circt-lint.yaml`:

```yaml
# Rule configurations
rules:
  # Severity: ignore, hint, warning, error
  unused_signal: warning
  unused_parameter: warning
  undriven_signal: error
  unread_signal: warning
  implicit_width: hint
  blocking_in_sequential: error
  nonblocking_in_combinational: warning
  missing_default: warning
  incomplete_case: warning
  multiple_drivers: error
  latch_inference: warning

  # Rule with options
  naming_convention:
    severity: warning
    enabled: true

# Naming convention patterns
naming:
  module_pattern: "^[A-Z][a-zA-Z0-9_]*$"
  signal_pattern: "^[a-z][a-z0-9_]*$"
  parameter_pattern: "^[A-Z][A-Z0-9_]*$"
  port_pattern: "^[a-z][a-z0-9_]*(_i|_o|_io)?$"
  instance_pattern: "^[a-z][a-z0-9_]*$"
  constant_pattern: "^[A-Z][A-Z0-9_]*$"

# File exclusions
exclude:
  - "generated/**"
  - "third_party/**"
  - "**/testbench/**"
```

### Project Integration

Reference lint configuration from `circt-project.yaml`:

```yaml
project:
  name: "my_design"

lint:
  enabled: true
  config: "lint.yaml"
  enable_rules:
    - "unused_signal"
  disable_rules:
    - "implicit_width"
  exclude_patterns:
    - "generated/**"
```

## Available Rules

### Structural Rules

| Rule | Description | Default |
|------|-------------|---------|
| `unused_signal` | Signal declared but never used | warning |
| `unused_parameter` | Parameter declared but never used | warning |
| `undriven_signal` | Signal used but never assigned | error |
| `unread_signal` | Signal assigned but never read | warning |
| `multiple_drivers` | Signal driven from multiple sources | error |
| `floating_input` | Input port left unconnected | warning |
| `floating_output` | Output port left unconnected | warning |

### Coding Style Rules

| Rule | Description | Default |
|------|-------------|---------|
| `blocking_in_sequential` | Blocking assignment in sequential block | error |
| `nonblocking_in_combinational` | Non-blocking in combinational block | warning |
| `implicit_width` | Implicit width in operations | hint |
| `missing_default` | Case without default clause | warning |
| `incomplete_case` | Case statement missing cases | warning |
| `latch_inference` | Unintentional latch inference | warning |
| `naming_convention` | Symbol naming violations | warning |

### Best Practice Rules

| Rule | Description | Default |
|------|-------------|---------|
| `magic_numbers` | Literal numbers without named constant | hint |
| `missing_ports` | Module instantiation with implicit ports | hint |
| `deprecated_syntax` | Use of deprecated language features | warning |
| `race_condition` | Potential race condition detected | error |

## Output Examples

### Text Output

```
error: undriven_signal
   --> rtl/counter.sv:10:5
    |
 10 |     logic [7:0] data;
    |     ^^^^^^^^^^^^ signal 'data' is never assigned
    |
    = help: assign a value to 'data' or remove it if unused

warning: blocking_in_sequential
   --> rtl/counter.sv:25:8
    |
 25 |         count = count + 1;
    |         ^^^^^^^^^^^^^^^^ blocking assignment in sequential block
    |
    = help: use non-blocking assignment (<=) in sequential blocks
```

### JUnit XML Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="circt-lint" tests="15" failures="2" errors="0">
  <testsuite name="rtl/counter.sv" tests="5" failures="1">
    <testcase name="undriven_signal" classname="lint.structural">
      <failure message="signal 'data' is never assigned">
        rtl/counter.sv:10:5
      </failure>
    </testcase>
  </testsuite>
</testsuites>
```

### SARIF Output

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "circt-lint",
        "rules": [...]
      }
    },
    "results": [...]
  }]
}
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run lint
  run: |
    circt-lint --format=sarif --output=results.sarif .

- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

See the [CI/CD Integration Guide](../ci-integration/README.md) for more examples.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No errors (warnings allowed) |
| 1 | Errors found |
| 2 | Configuration error |
| 3 | File not found |

## Suppressing Warnings

### In Code

```systemverilog
// lint-disable unused_signal
logic [7:0] spare_signal;
// lint-enable unused_signal

// lint-disable-next-line blocking_in_sequential
count = count + 1;

/* lint-disable naming_convention */
logic BadlyNamedSignal;
/* lint-enable naming_convention */
```

### In Configuration

```yaml
rules:
  unused_signal:
    enabled: false

exclude:
  - "generated/**"
```

## Custom Rules

Custom lint rules can be added by implementing the `LintRule` interface:

```cpp
class MyCustomRule : public LintRule {
public:
  StringRef getName() const override { return "my_custom_rule"; }
  StringRef getDescription() const override { return "My custom check"; }

  void check(const SourceFile &file, DiagnosticEmitter &diag) override {
    // Implement your check here
  }
};

// Register the rule
static LintRuleRegistration<MyCustomRule> reg;
```

## Performance

For large projects:

- Use `exclude` patterns to skip generated code
- Enable only necessary rules
- Use target-specific configurations to limit scope
- Consider incremental analysis for CI

## Related Documentation

- [circt-verilog-lsp-server](circt-verilog-lsp-server.md) - LSP server with integrated linting
- [CI/CD Integration](../ci-integration/README.md) - Pipeline integration guide
- [Project Configuration](../ci-integration/README.md#project-configuration) - Configuration format
