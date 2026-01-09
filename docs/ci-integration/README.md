# CI/CD Integration Guide for CIRCT Verilog Tools

This guide explains how to integrate CIRCT's Verilog/SystemVerilog tools into your CI/CD pipeline.

## Overview

CIRCT provides several tools that can be integrated into CI/CD workflows:

- **circt-verilog**: Verilog/SystemVerilog compiler frontend
- **circt-lint**: Configurable linting tool (when enabled)
- **circt-verilog-lsp-server**: Language server for IDE integration

## Output Formats

### JUnit XML

CIRCT tools can output test results in JUnit XML format, which is understood by most CI/CD systems:

```bash
circt-lint --junit-xml=results.xml .
```

The JUnit XML output includes:
- Test suites for each category of checks
- Individual test cases for each file/rule combination
- Failure messages with source locations
- Execution timing information

### SARIF (Static Analysis Results Interchange Format)

SARIF is supported for integration with GitHub Code Scanning and other security tools:

```bash
circt-lint --format=sarif --output=results.sarif .
```

Benefits of SARIF:
- Native integration with GitHub Security tab
- Rich location information with code snippets
- Support for suggested fixes
- Rule metadata and help URLs

### JSON

For custom tooling and analysis:

```bash
circt-lint --format=json --output=results.json .
```

## GitHub Actions Integration

### Basic Lint Workflow

Create `.github/workflows/verilog-lint.yml`:

```yaml
name: Verilog Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run lint
        run: |
          circt-lint --junit-xml=results.xml .

      - name: Publish results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: results.xml
```

### SARIF Upload for Code Scanning

```yaml
- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

This will:
- Show lint results in the Security tab
- Add annotations to changed files in PRs
- Block merging if configured

## Project Configuration

Create a `circt-project.yaml` in your repository root:

```yaml
project:
  name: "my_chip"
  top: "top_module"
  version: "1.0.0"

sources:
  include_dirs:
    - "rtl/"
    - "includes/"
  defines:
    - "SYNTHESIS"
  files:
    - "rtl/**/*.sv"
    - "rtl/**/*.v"

lint:
  enabled: true
  config: "lint.yaml"

targets:
  synthesis:
    top: "chip_top"
    defines: ["SYNTHESIS"]
  simulation:
    top: "tb_top"
    defines: ["SIMULATION"]
```

## Jenkins Integration

### Pipeline Example

```groovy
pipeline {
    agent any

    stages {
        stage('Lint') {
            steps {
                sh 'circt-lint --junit-xml=lint-results.xml .'
            }
            post {
                always {
                    junit 'lint-results.xml'
                }
            }
        }

        stage('Elaborate') {
            steps {
                sh 'circt-verilog --elaborate --top=top_module .'
            }
        }
    }
}
```

## GitLab CI Integration

```yaml
verilog-lint:
  stage: test
  script:
    - circt-lint --junit-xml=results.xml .
  artifacts:
    reports:
      junit: results.xml
    when: always
```

## Azure DevOps Integration

```yaml
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      circt-lint --junit-xml=$(System.DefaultWorkingDirectory)/results.xml .

- task: PublishTestResults@2
  inputs:
    testResultsFormat: 'JUnit'
    testResultsFiles: '**/results.xml'
```

## Best Practices

### 1. Cache CIRCT Installation

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.circt
    key: circt-${{ runner.os }}-${{ hashFiles('**/circt-project.yaml') }}
```

### 2. Run Lint on Changed Files Only

```yaml
- name: Get changed files
  id: changes
  uses: tj-actions/changed-files@v42
  with:
    files: |
      **/*.v
      **/*.sv

- name: Lint changed files
  if: steps.changes.outputs.any_changed == 'true'
  run: |
    circt-lint ${{ steps.changes.outputs.all_changed_files }}
```

### 3. Configure Required Checks

In GitHub repository settings:
1. Go to Settings > Branches > Branch protection rules
2. Add rule for `main` branch
3. Enable "Require status checks to pass"
4. Select "Verilog Lint" as required check

### 4. Use Matrix Builds for Multiple Configurations

```yaml
strategy:
  matrix:
    target: [synthesis, simulation]
steps:
  - run: circt-lint --target=${{ matrix.target }} .
```

## Troubleshooting

### Common Issues

1. **Missing include files**: Ensure `include_dirs` in `circt-project.yaml` is correct
2. **Undefined macros**: Add missing defines to the configuration
3. **SARIF upload fails**: Check file size limits and format validity

### Debug Mode

Enable verbose output for debugging:

```bash
circt-lint --verbose --log=debug .
```

## Example Workflows

See the `docs/ci-integration/` directory for complete workflow examples:

- `github-actions-lint.yml` - Full linting workflow with SARIF
- `github-actions-build.yml` - Elaboration and simulation setup
