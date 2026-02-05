"""Pytest configuration and shared fixtures for LSP stress tests."""

import asyncio
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest
import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities

# Paths
CIRCT_BUILD_DIR = pathlib.Path(__file__).parents[4] / "build"
CIRCT_VERILOG_LSP = CIRCT_BUILD_DIR / "bin" / "circt-verilog-lsp-server"
OPENTITAN_ROOT = pathlib.Path.home() / "opentitan"
OPENTITAN_HW = OPENTITAN_ROOT / "hw"

# Test directories
TEST_DIR = pathlib.Path(__file__).parent


@dataclass
class BenchmarkResult:
    """Result of a benchmark operation."""
    operation: str
    duration_ms: float
    memory_delta_mb: float = 0.0
    file_path: Optional[str] = None
    file_lines: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkCollector:
    """Collects benchmark results during test runs."""
    results: list = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> str:
        """Generate a summary of all benchmark results."""
        lines = ["=" * 60, "BENCHMARK SUMMARY", "=" * 60]

        # Group by operation
        by_op = {}
        for r in self.results:
            if r.operation not in by_op:
                by_op[r.operation] = []
            by_op[r.operation].append(r)

        for op, results in sorted(by_op.items()):
            successful = [r for r in results if r.success]
            if successful:
                avg_ms = sum(r.duration_ms for r in successful) / len(successful)
                max_ms = max(r.duration_ms for r in successful)
                min_ms = min(r.duration_ms for r in successful)
                lines.append(f"\n{op}:")
                lines.append(f"  Count: {len(successful)}/{len(results)}")
                lines.append(f"  Avg: {avg_ms:.1f}ms, Min: {min_ms:.1f}ms, Max: {max_ms:.1f}ms")

        lines.append("=" * 60)
        return "\n".join(lines)


@pytest.fixture(scope="session")
def benchmark_collector():
    """Session-wide benchmark collector."""
    collector = BenchmarkCollector()
    yield collector
    print("\n" + collector.summary())


@pytest.fixture
def benchmark(benchmark_collector):
    """Context manager for benchmarking operations."""
    class Benchmark:
        def __init__(self):
            self.start_time = None
            self.operation = None
            self.file_path = None
            self.file_lines = 0

        def start(self, operation: str, file_path: str = None, file_lines: int = 0):
            self.operation = operation
            self.file_path = file_path
            self.file_lines = file_lines
            self.start_time = time.perf_counter()

        def stop(self, success: bool = True, error: str = None):
            if self.start_time is None:
                return
            elapsed = (time.perf_counter() - self.start_time) * 1000
            result = BenchmarkResult(
                operation=self.operation,
                duration_ms=elapsed,
                file_path=self.file_path,
                file_lines=self.file_lines,
                success=success,
                error=error
            )
            benchmark_collector.add(result)
            self.start_time = None
            return result

    return Benchmark()


def get_opentitan_include_paths() -> list[str]:
    """Get include paths for OpenTitan workspace."""
    paths = []

    # IP RTL directories
    ip_dir = OPENTITAN_HW / "ip"
    if ip_dir.exists():
        for ip in ip_dir.iterdir():
            if ip.is_dir():
                rtl_dir = ip / "rtl"
                if rtl_dir.exists():
                    paths.append(str(rtl_dir))

    # Top-level generated RTL
    for top in ["top_earlgrey", "top_darjeeling", "top_englishbreakfast"]:
        top_dir = OPENTITAN_HW / top
        if top_dir.exists():
            rtl_autogen = top_dir / "rtl" / "autogen"
            if rtl_autogen.exists():
                paths.append(str(rtl_autogen))
            # IP autogen
            ip_autogen = top_dir / "ip_autogen"
            if ip_autogen.exists():
                for ip in ip_autogen.iterdir():
                    if ip.is_dir():
                        ip_rtl = ip / "rtl"
                        if ip_rtl.exists():
                            paths.append(str(ip_rtl))

    # DV libraries
    dv_sv = OPENTITAN_HW / "dv" / "sv"
    if dv_sv.exists():
        for lib in dv_sv.iterdir():
            if lib.is_dir():
                paths.append(str(lib))

    # Vendor libraries
    vendor = OPENTITAN_HW / "vendor"
    if vendor.exists():
        paths.append(str(vendor))

    return paths


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[str(CIRCT_VERILOG_LSP)],
    ),
)
async def lsp_client(lsp_client: LanguageClient):
    """Basic LSP client fixture for simple tests."""
    params = types.InitializeParams(
        capabilities=client_capabilities("visual-studio-code"),
        root_uri=TEST_DIR.as_uri(),
        workspace_folders=[
            types.WorkspaceFolder(uri=TEST_DIR.as_uri(), name="test"),
        ],
    )
    await lsp_client.initialize_session(params)
    yield lsp_client
    await lsp_client.shutdown_session()


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[str(CIRCT_VERILOG_LSP)],
    ),
)
async def opentitan_client(lsp_client: LanguageClient):
    """LSP client configured for OpenTitan workspace."""
    params = types.InitializeParams(
        capabilities=client_capabilities("visual-studio-code"),
        root_uri=OPENTITAN_HW.as_uri(),
        workspace_folders=[
            types.WorkspaceFolder(uri=OPENTITAN_HW.as_uri(), name="opentitan-hw"),
        ],
    )
    await lsp_client.initialize_session(params)
    yield lsp_client
    await lsp_client.shutdown_session()


async def open_document(client: LanguageClient, file_path: pathlib.Path, wait_time: float = 0.5):
    """Open a document in the LSP client."""
    uri = file_path.as_uri()
    content = file_path.read_text()

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="systemverilog",
                version=1,
                text=content,
            )
        )
    )

    await asyncio.sleep(wait_time)
    return uri, content


def count_file_lines(file_path: pathlib.Path) -> int:
    """Count lines in a file."""
    try:
        return sum(1 for _ in file_path.open())
    except Exception:
        return 0


# OpenTitan test file paths
class OpenTitanFiles:
    """Paths to key OpenTitan files for testing."""

    # Small, well-structured files
    UART_TX = OPENTITAN_HW / "ip" / "uart" / "rtl" / "uart_tx.sv"
    UART = OPENTITAN_HW / "ip" / "uart" / "rtl" / "uart.sv"
    UART_PKG = OPENTITAN_HW / "ip" / "uart" / "rtl" / "uart_reg_pkg.sv"

    # Complex RTL
    AES = OPENTITAN_HW / "ip" / "aes" / "rtl" / "aes.sv"
    AES_CORE = OPENTITAN_HW / "ip" / "aes" / "rtl" / "aes_core.sv"
    AES_PKG = OPENTITAN_HW / "ip" / "aes" / "rtl" / "aes_pkg.sv"
    KMAC = OPENTITAN_HW / "ip" / "kmac" / "rtl" / "kmac.sv"

    # Large auto-generated files
    PINMUX_REG_TOP = OPENTITAN_HW / "top_earlgrey" / "ip_autogen" / "pinmux" / "rtl" / "pinmux_reg_top.sv"
    ALERT_HANDLER_REG_TOP = OPENTITAN_HW / "top_earlgrey" / "ip_autogen" / "alert_handler" / "rtl" / "alert_handler_reg_top.sv"
    RV_PLIC_REG_TOP = OPENTITAN_HW / "top_earlgrey" / "ip_autogen" / "rv_plic" / "rtl" / "rv_plic_reg_top.sv"

    # Top-level
    TOP_EARLGREY = OPENTITAN_HW / "top_earlgrey" / "rtl" / "autogen" / "top_earlgrey.sv"

    # UVM/DV
    CIP_BASE_ENV = OPENTITAN_HW / "dv" / "sv" / "cip_lib" / "cip_base_env.sv"
    CIP_BASE_TEST = OPENTITAN_HW / "dv" / "sv" / "cip_lib" / "cip_base_test.sv"
    DV_BASE_TEST = OPENTITAN_HW / "dv" / "sv" / "dv_lib" / "dv_base_test.sv"
    DV_BASE_VSEQ = OPENTITAN_HW / "dv" / "sv" / "dv_lib" / "dv_base_vseq.sv"

    # TLUL interface
    TLUL_PKG = OPENTITAN_HW / "ip" / "tlul" / "rtl" / "tlul_pkg.sv"

    @classmethod
    def exists(cls, attr: str) -> bool:
        """Check if a file exists."""
        path = getattr(cls, attr, None)
        return path is not None and path.exists()


# Skip decorator for missing files
def skip_if_missing(file_attr: str):
    """Skip test if OpenTitan file is missing."""
    def decorator(func):
        path = getattr(OpenTitanFiles, file_attr, None)
        if path is None or not path.exists():
            return pytest.mark.skip(reason=f"OpenTitan file not found: {file_attr}")(func)
        return func
    return decorator
