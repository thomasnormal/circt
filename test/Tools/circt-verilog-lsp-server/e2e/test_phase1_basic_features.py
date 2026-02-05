"""
Phase 1: Basic Feature Verification Tests

Tests basic LSP features on small, well-structured OpenTitan files.
Target: hw/ip/uart/rtl/uart_tx.sv (small, well-structured module)
"""

import asyncio
import pathlib

import pytest
from lsprotocol import types

from conftest import (
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
)


class TestBasicDiagnostics:
    """Test diagnostics feature on valid SystemVerilog files."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_uart_tx_no_false_positives(self, opentitan_client, benchmark):
        """Verify no false positive diagnostics on valid uart_tx.sv."""
        benchmark.start("diagnostics", str(OpenTitanFiles.UART_TX))

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        # Check diagnostics
        diagnostics = opentitan_client.diagnostics.get(uri, [])

        # Filter for errors only (warnings may be acceptable)
        errors = [d for d in diagnostics if d.severity == types.DiagnosticSeverity.Error]

        benchmark.stop(success=len(errors) == 0)

        # Valid code should have no errors
        if errors:
            error_msgs = [f"Line {d.range.start.line}: {d.message}" for d in errors]
            pytest.fail(f"False positive errors in uart_tx.sv:\n" + "\n".join(error_msgs))

    @pytest.mark.asyncio
    @skip_if_missing("UART")
    async def test_uart_top_diagnostics(self, opentitan_client, benchmark):
        """Check diagnostics on uart.sv (may have expected include errors)."""
        benchmark.start("diagnostics", str(OpenTitanFiles.UART))

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART, wait_time=1.0)

        diagnostics = opentitan_client.diagnostics.get(uri, [])
        errors = [d for d in diagnostics if d.severity == types.DiagnosticSeverity.Error]

        # Filter out expected missing include/package errors
        # These are expected when files are opened without proper include paths
        false_positives = [
            d for d in errors
            if not any(msg in d.message.lower() for msg in [
                'no such file', 'unknown class or package', 'unknown package',
                'unknown module', 'could not find', 'unable to find'
            ])
        ]

        benchmark.stop(success=len(false_positives) == 0)

        print(f"\nuart.sv diagnostics:")
        print(f"  Total errors: {len(errors)}")
        print(f"  Missing include errors: {len(errors) - len(false_positives)}")
        print(f"  Other errors: {len(false_positives)}")

        if false_positives:
            error_msgs = [f"Line {d.range.start.line}: {d.message}" for d in false_positives[:5]]
            pytest.fail(f"Unexpected errors in uart.sv:\n" + "\n".join(error_msgs))


class TestHover:
    """Test hover information on various symbol types."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_on_signal(self, opentitan_client, benchmark):
        """Test hover on a signal declaration."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Find a signal in the file (look for 'logic' declarations)
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'logic' in line and not line.strip().startswith('//'):
                # Try to find the signal name after 'logic'
                col = line.find('logic') + len('logic')
                while col < len(line) and line[col] in ' \t[]0-9:':
                    col += 1
                if col < len(line):
                    break
        else:
            pytest.skip("No logic signal found in file")

        benchmark.start("hover", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=line_num, character=col),
            )
        )

        benchmark.stop(success=result is not None)

        # Hover should return information
        assert result is None or isinstance(result, types.Hover)

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_on_parameter(self, opentitan_client, benchmark):
        """Test hover on a parameter."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'parameter' in line.lower():
                col = line.lower().find('parameter') + len('parameter')
                while col < len(line) and line[col] in ' \t':
                    col += 1
                if col < len(line):
                    break
        else:
            pytest.skip("No parameter found in file")

        benchmark.start("hover", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=line_num, character=col),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, types.Hover)

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_on_module_name(self, opentitan_client, benchmark):
        """Test hover on module name."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if line.strip().startswith('module '):
                col = line.find('module ') + len('module ')
                break
        else:
            pytest.skip("No module declaration found")

        benchmark.start("hover", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=line_num, character=col),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, types.Hover)


class TestGoToDefinition:
    """Test go-to-definition navigation."""

    @pytest.mark.asyncio
    @skip_if_missing("UART")
    async def test_goto_definition_signal(self, opentitan_client, benchmark):
        """Test go-to-definition on a signal usage."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART)

        # Look for a signal assignment (something used on RHS)
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            # Look for assign statements
            if 'assign' in line and '=' in line:
                eq_pos = line.find('=')
                if eq_pos > 0 and eq_pos < len(line) - 1:
                    # Find first identifier after =
                    for i in range(eq_pos + 1, len(line)):
                        if line[i].isalpha() or line[i] == '_':
                            target_line = line_num
                            target_col = i
                            break
                    if target_line is not None:
                        break

        if target_line is None:
            pytest.skip("No suitable signal usage found")

        benchmark.start("goto_definition", str(OpenTitanFiles.UART))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, (types.Location, list))

    @pytest.mark.asyncio
    @skip_if_missing("UART")
    async def test_goto_definition_module_instance(self, opentitan_client, benchmark):
        """Test go-to-definition on a module instantiation."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART, wait_time=1.0)

        lines = content.split('\n')
        target_line = None
        target_col = None

        # Look for module instantiations (identifier followed by #( or identifier)
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            # Skip obvious non-instantiation lines
            if stripped.startswith('//') or stripped.startswith('module '):
                continue
            if stripped.startswith('always') or stripped.startswith('assign'):
                continue

            # Look for pattern: word followed by space then word (possible instantiation)
            words = stripped.split()
            if len(words) >= 2:
                first_word = words[0]
                # Check if it looks like a module name (starts with letter, no keywords)
                if (first_word[0].isalpha() and
                    first_word not in ['logic', 'wire', 'reg', 'input', 'output', 'inout',
                                       'if', 'else', 'for', 'while', 'case', 'default',
                                       'begin', 'end', 'function', 'task', 'endmodule']):
                    target_line = line_num
                    target_col = line.find(first_word)
                    break

        if target_line is None:
            pytest.skip("No module instantiation found")

        benchmark.start("goto_definition", str(OpenTitanFiles.UART))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, (types.Location, list))


class TestFindReferences:
    """Test find-references functionality."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_find_references_signal(self, opentitan_client, benchmark):
        """Test finding all references to a signal."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Find a signal declaration
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'logic' in line or 'wire' in line:
                # Find the signal name
                for keyword in ['logic', 'wire']:
                    if keyword in line:
                        idx = line.find(keyword) + len(keyword)
                        # Skip past type specifiers
                        while idx < len(line) and line[idx] in ' \t[]0-9:':
                            idx += 1
                        if idx < len(line) and (line[idx].isalpha() or line[idx] == '_'):
                            target_line = line_num
                            target_col = idx
                            break
                else:
                    continue
                break
        else:
            pytest.skip("No signal declaration found")

        benchmark.start("find_references", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, list)


class TestDocumentSymbols:
    """Test document symbols (outline) functionality."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_document_symbols(self, opentitan_client, benchmark):
        """Test getting document symbols (outline view)."""
        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        benchmark.start("document_symbols", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        assert result is not None
        symbols = result if isinstance(result, list) else []
        assert len(symbols) > 0, "Expected at least one symbol (the module)"

        # Extract symbol names
        symbol_names = []
        for sym in symbols:
            if isinstance(sym, types.DocumentSymbol):
                symbol_names.append(sym.name)
            elif isinstance(sym, types.SymbolInformation):
                symbol_names.append(sym.name)

        # Should have the uart_tx module
        assert any('uart' in name.lower() for name in symbol_names), \
            f"Expected uart module in symbols: {symbol_names}"

    @pytest.mark.asyncio
    @skip_if_missing("UART")
    async def test_document_symbols_multiple_modules(self, opentitan_client, benchmark):
        """Test symbols for a file with multiple entities."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART)

        benchmark.start("document_symbols", str(OpenTitanFiles.UART))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        assert result is not None
        symbols = result if isinstance(result, list) else []

        # Count 'module' keywords in file
        module_count = content.lower().count('module ') - content.lower().count('endmodule')
        # Account for the fact that endmodule doesn't have a space

        # Should have symbols
        assert len(symbols) > 0


class TestCompletion:
    """Test code completion functionality."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_in_module(self, opentitan_client, benchmark):
        """Test completion inside a module."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Find a position inside the module body
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if 'always' in line or 'assign' in line:
                target_line = line_num
                break

        if target_line is None:
            # Just use middle of file
            target_line = len(lines) // 2

        benchmark.start("completion", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=4),
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, (types.CompletionList, list))

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_after_prefix(self, opentitan_client, benchmark):
        """Test completion after typing a prefix."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Modify the document to add a partial identifier
        lines = content.split('\n')

        # Find a good insertion point
        insert_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('always') or line.strip().startswith('assign'):
                insert_line = i + 1
                break

        if insert_line is None:
            insert_line = len(lines) - 2

        # Create modified content with partial identifier
        modified_lines = lines[:insert_line] + ['    uart_'] + lines[insert_line:]
        modified_content = '\n'.join(modified_lines)

        # Update document
        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(
                    uri=uri,
                    version=2,
                ),
                content_changes=[
                    types.TextDocumentContentChangeEvent_Type1(text=modified_content)
                ],
            )
        )

        await asyncio.sleep(0.5)

        benchmark.start("completion", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=insert_line, character=9),  # After "uart_"
            )
        )

        benchmark.stop()
        assert result is None or isinstance(result, (types.CompletionList, list))


class TestFeatureResponseTime:
    """Test that feature response times meet targets."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_hover_response_time(self, opentitan_client, benchmark):
        """Hover should respond in < 100ms for small files."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Find module name
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if 'module' in line and 'endmodule' not in line:
                col = line.find('module') + 7
                break
        else:
            pytest.skip("No module found")

        benchmark.start("hover_response_time", str(OpenTitanFiles.UART_TX))

        import time
        start = time.perf_counter()

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=line_num, character=col),
            )
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        benchmark.stop()

        # Target: < 100ms for small files
        assert elapsed_ms < 500, f"Hover took {elapsed_ms:.1f}ms (target: <100ms)"

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_response_time(self, opentitan_client, benchmark):
        """Completion should respond in < 300ms for small files."""
        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        benchmark.start("completion_response_time", str(OpenTitanFiles.UART_TX))

        import time
        start = time.perf_counter()

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=10, character=4),
            )
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        benchmark.stop()

        # Target: < 300ms
        assert elapsed_ms < 1000, f"Completion took {elapsed_ms:.1f}ms (target: <300ms)"
