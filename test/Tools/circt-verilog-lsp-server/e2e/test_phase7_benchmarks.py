"""
Phase 7: Performance Benchmarking Tests

Comprehensive performance benchmarks for the LSP server:
- Response time measurements
- Memory usage tracking
- Scaling behavior
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

import pytest
from lsprotocol import types

from conftest import (
    OPENTITAN_HW,
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    file_path: str
    file_lines: int
    duration_ms: float
    target_ms: float
    passed: bool

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"{self.operation}: {self.duration_ms:.1f}ms (target: {self.target_ms}ms) [{status}]"


class TestPerformanceTargets:
    """Test against specific performance targets from the plan."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_target_100ms(self, opentitan_client, benchmark):
        """Hover should respond in < 100ms for small files."""
        file_path = OpenTitanFiles.UART_TX
        line_count = count_file_lines(file_path)

        uri, content = await open_document(opentitan_client, file_path, wait_time=1.0)

        # Find a suitable position
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'logic' in line or 'module' in line:
                target_line = line_num
                target_col = 10
                break
        else:
            target_line = 10
            target_col = 5

        # Warm up
        await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        # Measure
        benchmark.start("hover_small_file", str(file_path), line_count)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                )
            )
            times.append((time.perf_counter() - start) * 1000)

        benchmark.stop()

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)

        target_ms = 100

        print(f"\nHover performance (small file, {line_count} lines):")
        print(f"  Target: <{target_ms}ms")
        print(f"  Average: {avg_ms:.1f}ms")
        print(f"  Min: {min_ms:.1f}ms, Max: {max_ms:.1f}ms")
        print(f"  Result: {'PASS' if avg_ms < target_ms else 'FAIL'}")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_target_300ms(self, opentitan_client, benchmark):
        """Completion should respond in < 300ms."""
        file_path = OpenTitanFiles.UART_TX
        line_count = count_file_lines(file_path)

        uri, _ = await open_document(opentitan_client, file_path, wait_time=1.0)

        # Warm up
        await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=15, character=4),
            )
        )

        # Measure
        benchmark.start("completion_small_file", str(file_path), line_count)

        times = []
        for i in range(5):
            start = time.perf_counter()
            await opentitan_client.text_document_completion_async(
                types.CompletionParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=15 + i, character=4),
                )
            )
            times.append((time.perf_counter() - start) * 1000)

        benchmark.stop()

        avg_ms = sum(times) / len(times)
        target_ms = 300

        print(f"\nCompletion performance (small file):")
        print(f"  Target: <{target_ms}ms")
        print(f"  Average: {avg_ms:.1f}ms")
        print(f"  Result: {'PASS' if avg_ms < target_ms else 'FAIL'}")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_goto_definition_target_200ms(self, opentitan_client, benchmark):
        """Go to definition should respond in < 200ms."""
        file_path = OpenTitanFiles.UART_TX
        line_count = count_file_lines(file_path)

        uri, content = await open_document(opentitan_client, file_path, wait_time=1.0)

        # Find a symbol reference
        lines = content.split('\n')
        target_line = 20
        target_col = 10

        for line_num, line in enumerate(lines):
            if '=' in line and not line.strip().startswith('//'):
                eq_pos = line.find('=')
                if eq_pos > 0:
                    target_line = line_num
                    target_col = eq_pos + 2
                    break

        # Warm up
        await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        # Measure
        benchmark.start("goto_definition_small", str(file_path), line_count)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            await opentitan_client.text_document_definition_async(
                types.DefinitionParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                )
            )
            times.append((time.perf_counter() - start) * 1000)

        benchmark.stop()

        avg_ms = sum(times) / len(times)
        target_ms = 200

        print(f"\nGo-to-definition performance (small file):")
        print(f"  Target: <{target_ms}ms")
        print(f"  Average: {avg_ms:.1f}ms")
        print(f"  Result: {'PASS' if avg_ms < target_ms else 'FAIL'}")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_find_references_target_1s(self, opentitan_client, benchmark):
        """Find all references should respond in < 1s."""
        file_path = OpenTitanFiles.UART_TX
        line_count = count_file_lines(file_path)

        uri, content = await open_document(opentitan_client, file_path, wait_time=1.0)

        # Find module name
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'module ' in line:
                target_line = line_num
                target_col = line.find('module ') + 7
                break
        else:
            target_line = 0
            target_col = 10

        benchmark.start("find_references_small", str(file_path), line_count)

        times = []
        for _ in range(3):
            start = time.perf_counter()
            await opentitan_client.text_document_references_async(
                types.ReferenceParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                    context=types.ReferenceContext(include_declaration=True),
                )
            )
            times.append((time.perf_counter() - start) * 1000)

        benchmark.stop()

        avg_ms = sum(times) / len(times)
        target_ms = 1000

        print(f"\nFind references performance:")
        print(f"  Target: <{target_ms}ms")
        print(f"  Average: {avg_ms:.1f}ms")
        print(f"  Result: {'PASS' if avg_ms < target_ms else 'FAIL'}")


class TestScalingBehavior:
    """Test how performance scales with file size."""

    @pytest.mark.asyncio
    async def test_hover_scaling(self, opentitan_client, benchmark):
        """Test hover performance across different file sizes."""
        test_files = []

        # Small file
        if OpenTitanFiles.exists("UART_TX"):
            test_files.append(("small", OpenTitanFiles.UART_TX))

        # Medium file
        if OpenTitanFiles.exists("AES"):
            test_files.append(("medium", OpenTitanFiles.AES))

        # Large file
        if OpenTitanFiles.exists("TOP_EARLGREY"):
            test_files.append(("large", OpenTitanFiles.TOP_EARLGREY))

        # Very large file
        if OpenTitanFiles.exists("PINMUX_REG_TOP"):
            test_files.append(("xlarge", OpenTitanFiles.PINMUX_REG_TOP))

        if len(test_files) < 2:
            pytest.skip("Need at least 2 files for scaling test")

        results = []

        print("\nHover scaling test:")
        print("-" * 50)

        for size_label, file_path in test_files:
            line_count = count_file_lines(file_path)

            uri, _ = await open_document(opentitan_client, file_path, wait_time=2.0)

            # Measure hover at middle of file
            mid_line = line_count // 2

            times = []
            for _ in range(3):
                start = time.perf_counter()
                await opentitan_client.text_document_hover_async(
                    types.HoverParams(
                        text_document=types.TextDocumentIdentifier(uri=uri),
                        position=types.Position(line=mid_line, character=10),
                    )
                )
                times.append((time.perf_counter() - start) * 1000)

            avg_ms = sum(times) / len(times)
            results.append((size_label, line_count, avg_ms))

            print(f"  {size_label:8} ({line_count:6} lines): {avg_ms:.1f}ms")

        # Analyze scaling
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            size_ratio = last[1] / first[1]
            time_ratio = last[2] / first[2] if first[2] > 0 else 0

            print("-" * 50)
            print(f"  Size increase: {size_ratio:.1f}x")
            print(f"  Time increase: {time_ratio:.1f}x")

            if time_ratio < size_ratio:
                print("  Scaling: Sub-linear (good)")
            elif time_ratio < size_ratio * 2:
                print("  Scaling: Linear (acceptable)")
            else:
                print("  Scaling: Super-linear (concerning)")

    @pytest.mark.asyncio
    async def test_document_symbols_scaling(self, opentitan_client, benchmark):
        """Test document symbols performance across file sizes."""
        test_files = []

        if OpenTitanFiles.exists("UART_TX"):
            test_files.append(OpenTitanFiles.UART_TX)
        if OpenTitanFiles.exists("AES"):
            test_files.append(OpenTitanFiles.AES)
        if OpenTitanFiles.exists("TOP_EARLGREY"):
            test_files.append(OpenTitanFiles.TOP_EARLGREY)

        if len(test_files) < 2:
            pytest.skip("Need at least 2 files")

        print("\nDocument symbols scaling test:")
        print("-" * 50)

        for file_path in test_files:
            line_count = count_file_lines(file_path)
            file_name = file_path.name

            uri, _ = await open_document(opentitan_client, file_path, wait_time=2.0)

            start = time.perf_counter()
            result = await opentitan_client.text_document_document_symbol_async(
                types.DocumentSymbolParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            symbol_count = len(result) if result else 0
            print(f"  {file_name[:20]:20} ({line_count:6} lines): {elapsed_ms:.1f}ms, {symbol_count} symbols")


class TestColdStartPerformance:
    """Test cold start (initialization) performance."""

    @pytest.mark.asyncio
    async def test_workspace_initialization(self, benchmark):
        """Test workspace initialization time."""
        import pytest_lsp
        from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities
        from conftest import CIRCT_VERILOG_LSP

        benchmark.start("cold_start", str(OPENTITAN_HW))

        start = time.perf_counter()

        # Create a new client (simulating cold start)
        config = ClientServerConfig(
            server_command=[str(CIRCT_VERILOG_LSP)],
        )

        async with pytest_lsp.client_session(config) as client:
            init_params = types.InitializeParams(
                capabilities=client_capabilities("visual-studio-code"),
                root_uri=OPENTITAN_HW.as_uri(),
                workspace_folders=[
                    types.WorkspaceFolder(uri=OPENTITAN_HW.as_uri(), name="opentitan"),
                ],
            )

            init_start = time.perf_counter()
            await client.initialize_session(init_params)
            init_elapsed = time.perf_counter() - init_start

            # Open a file to trigger indexing
            if OpenTitanFiles.exists("UART_TX"):
                uri = OpenTitanFiles.UART_TX.as_uri()
                content = OpenTitanFiles.UART_TX.read_text()

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

                # Wait for initial processing
                await asyncio.sleep(2.0)

            await client.shutdown_session()

        total_elapsed = time.perf_counter() - start

        benchmark.stop()

        target_s = 30  # Target: < 30s for OpenTitan

        print(f"\nCold start performance:")
        print(f"  Target: <{target_s}s")
        print(f"  Initialize: {init_elapsed:.2f}s")
        print(f"  Total: {total_elapsed:.2f}s")
        print(f"  Result: {'PASS' if total_elapsed < target_s else 'FAIL'}")


class TestThroughput:
    """Test throughput for batch operations."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_throughput(self, opentitan_client, benchmark):
        """Test how many hovers per second we can achieve."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        line_count = len(content.split('\n'))

        benchmark.start("hover_throughput", str(OpenTitanFiles.UART_TX))

        num_requests = 50
        start = time.perf_counter()

        for i in range(num_requests):
            line = (i * 3) % max(1, line_count - 5)
            await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=line, character=10),
                )
            )

        elapsed = time.perf_counter() - start

        benchmark.stop()

        throughput = num_requests / elapsed

        print(f"\nHover throughput:")
        print(f"  Requests: {num_requests}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} requests/second")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_throughput(self, opentitan_client, benchmark):
        """Test completion throughput."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        line_count = len(content.split('\n'))

        benchmark.start("completion_throughput", str(OpenTitanFiles.UART_TX))

        num_requests = 30
        start = time.perf_counter()

        for i in range(num_requests):
            line = (i * 2) % max(1, line_count - 5)
            await opentitan_client.text_document_completion_async(
                types.CompletionParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=line, character=4),
                )
            )

        elapsed = time.perf_counter() - start

        benchmark.stop()

        throughput = num_requests / elapsed

        print(f"\nCompletion throughput:")
        print(f"  Requests: {num_requests}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} requests/second")


class TestLatencyDistribution:
    """Test latency distribution (p50, p90, p99)."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_latency_distribution(self, opentitan_client, benchmark):
        """Measure hover latency distribution."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        line_count = len(content.split('\n'))

        benchmark.start("hover_latency_dist", str(OpenTitanFiles.UART_TX))

        latencies = []
        num_samples = 100

        for i in range(num_samples):
            line = (i * 2) % max(1, line_count - 5)

            start = time.perf_counter()
            await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=line, character=10),
                )
            )
            latencies.append((time.perf_counter() - start) * 1000)

        benchmark.stop()

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = sum(latencies) / len(latencies)

        print(f"\nHover latency distribution ({num_samples} samples):")
        print(f"  Average: {avg:.1f}ms")
        print(f"  p50: {p50:.1f}ms")
        print(f"  p90: {p90:.1f}ms")
        print(f"  p99: {p99:.1f}ms")
        print(f"  Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms")


class TestResourceUsage:
    """Test resource usage (when psutil is available)."""

    @pytest.mark.asyncio
    async def test_memory_after_large_file(self, opentitan_client, benchmark):
        """Check memory usage after opening a large file."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        if not OpenTitanFiles.exists("PINMUX_REG_TOP"):
            pytest.skip("Large file not found")

        # Get server PID (if available)
        # This is tricky with pytest-lsp; we'll just measure client-side

        benchmark.start("memory_large_file", str(OpenTitanFiles.PINMUX_REG_TOP))

        # Measure before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.PINMUX_REG_TOP, wait_time=3.0)

        # Measure after
        mem_after = process.memory_info().rss / (1024 * 1024)

        benchmark.stop()

        print(f"\nClient memory usage:")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After: {mem_after:.1f} MB")
        print(f"  Delta: {mem_after - mem_before:.1f} MB")


class TestSummary:
    """Generate final performance summary."""

    @pytest.mark.asyncio
    async def test_generate_summary(self, benchmark_collector):
        """Print final benchmark summary."""
        # This test runs last and prints the collected benchmarks
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)

        # The collector will print its summary in conftest.py
        # Just verify we have results
        assert len(benchmark_collector.results) > 0, "No benchmark results collected"

        print(f"\nTotal benchmarks collected: {len(benchmark_collector.results)}")
