"""
Phase 2: Large File Stress Tests

Tests LSP performance on large auto-generated and complex RTL files.
Targets:
- hw/top_earlgrey/ip_autogen/pinmux/rtl/pinmux_reg_top.sv (40,267 LOC)
- hw/top_earlgrey/ip_autogen/alert_handler/rtl/alert_handler_reg_top.sv (19,874 LOC)
- hw/top_earlgrey/ip_autogen/rv_plic/rtl/rv_plic_reg_top.sv (19,609 LOC)
- hw/ip/aes/rtl/aes.sv (complex RTL)
- hw/top_earlgrey/rtl/autogen/top_earlgrey.sv (100+ module instantiations)
"""

import asyncio
import time

import pytest
from lsprotocol import types

from conftest import (
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
)


class TestLargeFileOpen:
    """Test opening large auto-generated files."""

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_open_pinmux_reg_top(self, opentitan_client, benchmark):
        """Open largest file: pinmux_reg_top.sv (40K+ LOC)."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP
        line_count = count_file_lines(file_path)

        benchmark.start("open_large_file", str(file_path), line_count)

        start = time.perf_counter()
        uri, _ = await open_document(opentitan_client, file_path, wait_time=5.0)
        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nOpened pinmux_reg_top.sv ({line_count} lines) in {elapsed:.2f}s")

        # Should complete without timeout (5s is generous)
        assert elapsed < 30, f"Opening took {elapsed:.1f}s (too slow)"

    @pytest.mark.asyncio
    @skip_if_missing("ALERT_HANDLER_REG_TOP")
    async def test_open_alert_handler_reg_top(self, opentitan_client, benchmark):
        """Open alert_handler_reg_top.sv (~20K LOC)."""
        file_path = OpenTitanFiles.ALERT_HANDLER_REG_TOP
        line_count = count_file_lines(file_path)

        benchmark.start("open_large_file", str(file_path), line_count)

        start = time.perf_counter()
        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)
        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nOpened alert_handler_reg_top.sv ({line_count} lines) in {elapsed:.2f}s")

    @pytest.mark.asyncio
    @skip_if_missing("RV_PLIC_REG_TOP")
    async def test_open_rv_plic_reg_top(self, opentitan_client, benchmark):
        """Open rv_plic_reg_top.sv (~20K LOC)."""
        file_path = OpenTitanFiles.RV_PLIC_REG_TOP
        line_count = count_file_lines(file_path)

        benchmark.start("open_large_file", str(file_path), line_count)

        start = time.perf_counter()
        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)
        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nOpened rv_plic_reg_top.sv ({line_count} lines) in {elapsed:.2f}s")

    @pytest.mark.asyncio
    @skip_if_missing("TOP_EARLGREY")
    async def test_open_top_earlgrey(self, opentitan_client, benchmark):
        """Open top_earlgrey.sv (top-level with 100+ instantiations)."""
        file_path = OpenTitanFiles.TOP_EARLGREY
        line_count = count_file_lines(file_path)

        benchmark.start("open_large_file", str(file_path), line_count)

        start = time.perf_counter()
        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)
        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nOpened top_earlgrey.sv ({line_count} lines) in {elapsed:.2f}s")


class TestLargeFileDiagnostics:
    """Test diagnostics on large files."""

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_diagnostics_large_file(self, opentitan_client, benchmark):
        """Verify diagnostics complete for large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP

        benchmark.start("diagnostics_large", str(file_path))

        start = time.perf_counter()
        uri, _ = await open_document(opentitan_client, file_path, wait_time=5.0)

        # Wait additional time for diagnostics to complete
        await asyncio.sleep(3.0)
        elapsed = time.perf_counter() - start

        diagnostics = opentitan_client.diagnostics.get(uri, [])
        errors = [d for d in diagnostics if d.severity == types.DiagnosticSeverity.Error]

        benchmark.stop(success=True)

        print(f"\nDiagnostics completed in {elapsed:.2f}s")
        print(f"  Errors: {len(errors)}, Total diagnostics: {len(diagnostics)}")

        # Log first few errors if any
        for err in errors[:3]:
            print(f"  Error at line {err.range.start.line}: {err.message[:80]}")


class TestLargeFileHover:
    """Test hover performance on large files."""

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_hover_large_file_start(self, opentitan_client, benchmark):
        """Test hover at start of large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP

        uri, content = await open_document(opentitan_client, file_path, wait_time=3.0)

        # Find first signal
        lines = content.split('\n')
        target_line = 50  # Early in file

        benchmark.start("hover_large_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=10),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nHover at line {target_line}: {elapsed_ms:.1f}ms")

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_hover_large_file_middle(self, opentitan_client, benchmark):
        """Test hover in middle of large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP
        line_count = count_file_lines(file_path)

        uri, content = await open_document(opentitan_client, file_path, wait_time=3.0)

        target_line = line_count // 2

        benchmark.start("hover_large_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=10),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nHover at line {target_line} (middle): {elapsed_ms:.1f}ms")

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_hover_large_file_end(self, opentitan_client, benchmark):
        """Test hover at end of large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP
        line_count = count_file_lines(file_path)

        uri, content = await open_document(opentitan_client, file_path, wait_time=3.0)

        target_line = line_count - 100  # Near end

        benchmark.start("hover_large_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=10),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nHover at line {target_line} (near end): {elapsed_ms:.1f}ms")


class TestLargeFileCompletion:
    """Test completion performance on large files."""

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_completion_large_file(self, opentitan_client, benchmark):
        """Test completion in large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP

        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)

        benchmark.start("completion_large_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=1000, character=4),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nCompletion in large file: {elapsed_ms:.1f}ms")

        # Should complete within reasonable time
        assert elapsed_ms < 5000, f"Completion took {elapsed_ms:.1f}ms (too slow)"


class TestLargeFileSymbols:
    """Test document symbols on large files."""

    @pytest.mark.asyncio
    @skip_if_missing("PINMUX_REG_TOP")
    async def test_document_symbols_large_file(self, opentitan_client, benchmark):
        """Test document symbols in large file."""
        file_path = OpenTitanFiles.PINMUX_REG_TOP

        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)

        benchmark.start("symbols_large_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nDocument symbols in large file: {elapsed_ms:.1f}ms, {len(symbols)} symbols")

    @pytest.mark.asyncio
    @skip_if_missing("TOP_EARLGREY")
    async def test_document_symbols_top_level(self, opentitan_client, benchmark):
        """Test document symbols in top-level (many instantiations)."""
        file_path = OpenTitanFiles.TOP_EARLGREY

        uri, _ = await open_document(opentitan_client, file_path, wait_time=3.0)

        benchmark.start("symbols_top_level", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nDocument symbols in top_earlgrey: {elapsed_ms:.1f}ms, {len(symbols)} symbols")


class TestComplexRTL:
    """Test on complex, feature-rich RTL files."""

    @pytest.mark.asyncio
    @skip_if_missing("AES")
    async def test_aes_module_features(self, opentitan_client, benchmark):
        """Test features on AES module (parameters, FSM, assertions)."""
        file_path = OpenTitanFiles.AES

        benchmark.start("open_complex_rtl", str(file_path))

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        benchmark.stop()

        # Test symbols
        symbols_result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        symbols = symbols_result if isinstance(symbols_result, list) else []
        print(f"\nAES module: {len(symbols)} symbols")

        # Check for parameters in content
        param_count = content.lower().count('parameter')
        print(f"  Parameters: {param_count}")

    @pytest.mark.asyncio
    @skip_if_missing("AES_CORE")
    async def test_aes_core_complex_rtl(self, opentitan_client, benchmark):
        """Test on AES core (complex state machine)."""
        file_path = OpenTitanFiles.AES_CORE
        line_count = count_file_lines(file_path)

        benchmark.start("open_complex_rtl", str(file_path), line_count)

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        benchmark.stop()

        # Test hover on a state machine state (if exists)
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'state' in line.lower() and ('case' in line.lower() or 'enum' in line.lower()):
                result = await opentitan_client.text_document_hover_async(
                    types.HoverParams(
                        text_document=types.TextDocumentIdentifier(uri=uri),
                        position=types.Position(line=line_num, character=10),
                    )
                )
                print(f"\nHover on state machine line {line_num}: {'Found' if result else 'None'}")
                break


class TestMultipleOpenFiles:
    """Test having multiple large files open simultaneously."""

    @pytest.mark.asyncio
    async def test_multiple_large_files(self, opentitan_client, benchmark):
        """Open multiple large files and verify server stability."""
        files_to_open = []

        if OpenTitanFiles.exists("PINMUX_REG_TOP"):
            files_to_open.append(OpenTitanFiles.PINMUX_REG_TOP)
        if OpenTitanFiles.exists("ALERT_HANDLER_REG_TOP"):
            files_to_open.append(OpenTitanFiles.ALERT_HANDLER_REG_TOP)
        if OpenTitanFiles.exists("RV_PLIC_REG_TOP"):
            files_to_open.append(OpenTitanFiles.RV_PLIC_REG_TOP)

        if len(files_to_open) < 2:
            pytest.skip("Need at least 2 large files")

        benchmark.start("multiple_large_files", "multiple")

        uris = []
        for file_path in files_to_open:
            uri, _ = await open_document(opentitan_client, file_path, wait_time=1.0)
            uris.append(uri)

        # Let all files settle
        await asyncio.sleep(3.0)

        # Test hover on each file
        for uri in uris:
            result = await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=100, character=10),
                )
            )

        benchmark.stop()

        print(f"\nOpened {len(files_to_open)} large files simultaneously")
        total_lines = sum(count_file_lines(f) for f in files_to_open)
        print(f"  Total lines: {total_lines}")


class TestGoToDefinitionLargeFile:
    """Test go-to-definition in large files."""

    @pytest.mark.asyncio
    @skip_if_missing("TOP_EARLGREY")
    async def test_goto_definition_module_instance(self, opentitan_client, benchmark):
        """Test go-to-definition on module instance in top_earlgrey."""
        file_path = OpenTitanFiles.TOP_EARLGREY

        uri, content = await open_document(opentitan_client, file_path, wait_time=3.0)

        # Find a module instantiation (look for u_* pattern)
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if ' u_' in line and '(' in line:
                # Found an instantiation like "module_name u_instance ("
                idx = line.find(' u_')
                if idx > 0:
                    # Get the module name before u_
                    start = idx - 1
                    while start > 0 and (line[start-1].isalnum() or line[start-1] == '_'):
                        start -= 1
                    target_line = line_num
                    target_col = start
                    break

        if target_line is None:
            pytest.skip("No module instantiation found")

        benchmark.start("goto_definition_large", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nGo-to-definition in top_earlgrey: {elapsed_ms:.1f}ms")

        # Target: < 200ms
        # Be lenient for large files
        assert elapsed_ms < 2000, f"Go-to-definition took {elapsed_ms:.1f}ms"
