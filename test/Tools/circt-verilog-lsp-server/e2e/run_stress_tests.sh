#!/bin/bash
# Run LSP stress tests for CIRCT Verilog LSP Server
#
# Usage:
#   ./run_stress_tests.sh           # Run all tests
#   ./run_stress_tests.sh phase1    # Run specific phase
#   ./run_stress_tests.sh -v        # Verbose output
#   ./run_stress_tests.sh --quick   # Quick test run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CIRCT Verilog LSP Server Stress Tests${NC}"
echo -e "${BLUE}========================================${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi
    echo "  Python: $(python3 --version)"

    # Check LSP server
    LSP_SERVER="$SCRIPT_DIR/../../../../build/bin/circt-verilog-lsp-server"
    if [ ! -f "$LSP_SERVER" ]; then
        echo -e "${RED}Error: LSP server not found at $LSP_SERVER${NC}"
        echo "  Please build CIRCT first with: ninja circt-verilog-lsp-server"
        exit 1
    fi
    echo "  LSP Server: Found"

    # Check OpenTitan
    OPENTITAN_HW="$HOME/opentitan/hw"
    if [ -d "$OPENTITAN_HW" ]; then
        echo "  OpenTitan: Found at $OPENTITAN_HW"
    else
        echo -e "${YELLOW}  OpenTitan: Not found (some tests will be skipped)${NC}"
    fi

    # Check/create virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "\n${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
    fi

    # Activate and install dependencies
    source "$VENV_DIR/bin/activate"

    if ! python3 -c "import pytest_lsp" 2>/dev/null; then
        echo -e "\n${YELLOW}Installing dependencies...${NC}"
        pip install --quiet pytest pytest-asyncio pytest-lsp lsprotocol pygls cattrs
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Run tests
run_tests() {
    source "$VENV_DIR/bin/activate"
    cd "$SCRIPT_DIR"

    PYTEST_ARGS="-v --tb=short"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            phase1)
                PYTEST_ARGS="$PYTEST_ARGS test_phase1_basic_features.py"
                shift
                ;;
            phase2)
                PYTEST_ARGS="$PYTEST_ARGS test_phase2_large_files.py"
                shift
                ;;
            phase3)
                PYTEST_ARGS="$PYTEST_ARGS test_phase3_cross_file.py"
                shift
                ;;
            phase4)
                PYTEST_ARGS="$PYTEST_ARGS test_phase4_uvm.py"
                shift
                ;;
            phase5)
                PYTEST_ARGS="$PYTEST_ARGS test_phase5_rapid_editing.py"
                shift
                ;;
            phase6)
                PYTEST_ARGS="$PYTEST_ARGS test_phase6_feature_gaps.py"
                shift
                ;;
            phase7)
                PYTEST_ARGS="$PYTEST_ARGS test_phase7_benchmarks.py"
                shift
                ;;
            led)
                PYTEST_ARGS="$PYTEST_ARGS test_mini_project_led_ctrl.py"
                shift
                ;;
            basic)
                PYTEST_ARGS="$PYTEST_ARGS test_circt_verilog_lsp.py"
                shift
                ;;
            -v|--verbose)
                PYTEST_ARGS="$PYTEST_ARGS -v"
                shift
                ;;
            -vv)
                PYTEST_ARGS="$PYTEST_ARGS -vv"
                shift
                ;;
            --quick)
                PYTEST_ARGS="$PYTEST_ARGS -x --timeout=30"
                shift
                ;;
            --benchmark)
                PYTEST_ARGS="$PYTEST_ARGS test_phase7_benchmarks.py -v -s"
                shift
                ;;
            -k)
                PYTEST_ARGS="$PYTEST_ARGS -k $2"
                shift 2
                ;;
            *)
                PYTEST_ARGS="$PYTEST_ARGS $1"
                shift
                ;;
        esac
    done

    echo -e "\n${BLUE}Running tests...${NC}"
    echo "  pytest $PYTEST_ARGS"
    echo ""

    python3 -m pytest $PYTEST_ARGS
}

# Main
check_prerequisites
run_tests "$@"

echo -e "\n${GREEN}Done!${NC}"
