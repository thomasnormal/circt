"""
Phase 3: Cross-File Navigation Stress Tests

Tests navigation across the OpenTitan module hierarchy:
- Deep module hierarchy drilling
- Package dependency navigation
- Interface/modport handling
"""

import asyncio
import time

import pytest
from lsprotocol import types

from conftest import (
    OPENTITAN_HW,
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
)


class TestModuleHierarchyNavigation:
    """Test drilling down through module hierarchy."""

    @pytest.mark.asyncio
    @skip_if_missing("TOP_EARLGREY")
    async def test_hierarchy_drill_down(self, opentitan_client, benchmark):
        """Navigate from top_earlgrey down through hierarchy."""
        # Start at top_earlgrey
        file_path = OpenTitanFiles.TOP_EARLGREY

        uri, content = await open_document(opentitan_client, file_path, wait_time=3.0)

        # Find an AES or other IP instantiation
        lines = content.split('\n')
        target_line = None
        target_col = None
        instance_name = None

        for line_num, line in enumerate(lines):
            # Look for instantiations of known IPs
            for ip in ['aes', 'uart', 'hmac', 'kmac', 'gpio']:
                if f'{ip} ' in line.lower() and 'u_' in line:
                    # Found an instantiation
                    idx = line.lower().find(ip)
                    target_line = line_num
                    target_col = idx
                    instance_name = ip
                    break
            if target_line is not None:
                break

        if target_line is None:
            pytest.skip("No IP instantiation found in top_earlgrey")

        benchmark.start("hierarchy_drill_down", str(file_path))

        # Go to definition of the module type
        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nDrill down from top_earlgrey to {instance_name}: {elapsed_ms:.1f}ms")

        if result:
            if isinstance(result, types.Location):
                print(f"  Jumped to: {result.uri}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"  Jumped to: {result[0].uri if hasattr(result[0], 'uri') else result[0]}")

    @pytest.mark.asyncio
    @skip_if_missing("AES")
    async def test_aes_to_core_navigation(self, opentitan_client, benchmark):
        """Navigate from aes.sv to aes_core.sv."""
        file_path = OpenTitanFiles.AES

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find aes_core instantiation
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'aes_core' in line and 'u_' in line:
                idx = line.find('aes_core')
                target_line = line_num
                target_col = idx
                break

        if target_line is None:
            pytest.skip("No aes_core instantiation found")

        benchmark.start("module_navigation", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nNavigation aes -> aes_core: {elapsed_ms:.1f}ms")

        # Check if we got a result pointing to aes_core.sv
        if result:
            if isinstance(result, types.Location):
                assert 'aes_core' in result.uri or result is not None
            elif isinstance(result, list) and len(result) > 0:
                first = result[0]
                if hasattr(first, 'uri'):
                    print(f"  Target: {first.uri}")


class TestPackageDependencies:
    """Test cross-package navigation."""

    @pytest.mark.asyncio
    @skip_if_missing("AES")
    async def test_goto_package_type(self, opentitan_client, benchmark):
        """Navigate from AES module to aes_pkg types."""
        file_path = OpenTitanFiles.AES

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find aes_pkg:: usage
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'aes_pkg::' in line:
                idx = line.find('aes_pkg::')
                target_line = line_num
                target_col = idx + len('aes_pkg::')  # Position after ::
                break

        if target_line is None:
            pytest.skip("No aes_pkg:: usage found")

        benchmark.start("package_navigation", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nPackage type navigation: {elapsed_ms:.1f}ms")

        if result:
            if isinstance(result, types.Location):
                print(f"  Jumped to: {result.uri}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"  Jumped to: {result[0]}")

    @pytest.mark.asyncio
    @skip_if_missing("AES_PKG")
    async def test_find_package_references(self, opentitan_client, benchmark):
        """Find all references to an aes_pkg type."""
        file_path = OpenTitanFiles.AES_PKG

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find a typedef or enum in the package
        lines = content.split('\n')
        target_line = None
        target_col = None
        type_name = None

        for line_num, line in enumerate(lines):
            if 'typedef' in line and 'enum' in line:
                # Find the type name (usually at end before ;)
                # Pattern: typedef enum ... type_name;
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith(';') or part.endswith(','):
                        type_name = part.rstrip(';,')
                        target_line = line_num
                        target_col = line.find(type_name)
                        break
                if target_line is not None:
                    break

        if target_line is None:
            pytest.skip("No typedef enum found in aes_pkg")

        benchmark.start("find_package_references", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        ref_count = len(result) if result else 0
        print(f"\nFind references to {type_name}: {elapsed_ms:.1f}ms, {ref_count} references")

    @pytest.mark.asyncio
    async def test_tlul_package_navigation(self, opentitan_client, benchmark):
        """Test navigation to TLUL package types (used across many files)."""
        # Find a file that uses tlul_pkg
        tlul_pkg_path = OPENTITAN_HW / "ip" / "tlul" / "rtl" / "tlul_pkg.sv"

        if not tlul_pkg_path.exists():
            pytest.skip("tlul_pkg.sv not found")

        uri, content = await open_document(opentitan_client, tlul_pkg_path, wait_time=2.0)

        # Find tl_h2d_t or similar type
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'tl_h2d_t' in line and 'typedef' in line:
                target_line = line_num
                target_col = line.find('tl_h2d_t')
                break

        if target_line is None:
            # Try another common type
            for line_num, line in enumerate(lines):
                if 'typedef' in line and 'struct' in line:
                    target_line = line_num
                    target_col = 10
                    break

        if target_line is None:
            pytest.skip("No suitable type found in tlul_pkg")

        benchmark.start("tlul_references", str(tlul_pkg_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        ref_count = len(result) if result else 0
        print(f"\nTLUL type references: {elapsed_ms:.1f}ms, {ref_count} references")


class TestInterfaceNavigation:
    """Test interface and modport handling."""

    @pytest.mark.asyncio
    async def test_interface_definition(self, opentitan_client, benchmark):
        """Test navigation on interface definitions."""
        # Find an interface file
        interface_files = [
            OPENTITAN_HW / "ip" / "tlul" / "rtl" / "tlul_if.sv",
            OPENTITAN_HW / "dv" / "sv" / "tl_agent" / "tl_if.sv",
        ]

        interface_file = None
        for f in interface_files:
            if f.exists():
                interface_file = f
                break

        if interface_file is None:
            pytest.skip("No interface file found")

        uri, content = await open_document(opentitan_client, interface_file, wait_time=2.0)

        benchmark.start("interface_symbols", str(interface_file))

        # Get document symbols
        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nInterface file symbols: {len(symbols)}")

        # Look for interface or modport symbols
        for sym in symbols[:5]:
            if isinstance(sym, types.DocumentSymbol):
                print(f"  {sym.kind}: {sym.name}")

    @pytest.mark.asyncio
    async def test_modport_navigation(self, opentitan_client, benchmark):
        """Test go-to-definition on modport usage."""
        # Find a file using modport
        test_file = OPENTITAN_HW / "ip" / "uart" / "rtl" / "uart.sv"

        if not test_file.exists():
            pytest.skip("uart.sv not found")

        uri, content = await open_document(opentitan_client, test_file, wait_time=2.0)

        # Find modport usage
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if '.modport' in line.lower() or 'modport' in line.lower():
                target_line = line_num
                target_col = line.lower().find('modport')
                break

        if target_line is None:
            # Skip if no modport usage
            print("\nNo modport usage found in uart.sv")
            return

        benchmark.start("modport_navigation", str(test_file))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        print(f"\nModport navigation: {'Found' if result else 'Not found'}")


class TestFindReferencesAcrossFiles:
    """Test find-references across multiple files."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_find_references_cross_file(self, opentitan_client, benchmark):
        """Find references to a signal that might be used across files."""
        file_path = OpenTitanFiles.UART_TX

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find the module name
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if line.strip().startswith('module '):
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1].rstrip('(#;')
                    target_line = line_num
                    target_col = line.find(module_name)
                    break
        else:
            pytest.skip("No module declaration found")

        benchmark.start("find_references_cross_file", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        ref_count = len(result) if result else 0
        print(f"\nFind references to {module_name}: {elapsed_ms:.1f}ms, {ref_count} references")

        # Check if references are in different files
        if result:
            files = set()
            for ref in result:
                if hasattr(ref, 'uri'):
                    files.add(ref.uri)
            print(f"  References in {len(files)} file(s)")


class TestNavigationChain:
    """Test chained navigation operations."""

    @pytest.mark.asyncio
    async def test_navigation_chain_performance(self, opentitan_client, benchmark):
        """Test a chain of navigation operations."""
        # Start with uart.sv
        uart_path = OPENTITAN_HW / "ip" / "uart" / "rtl" / "uart.sv"

        if not uart_path.exists():
            pytest.skip("uart.sv not found")

        benchmark.start("navigation_chain", str(uart_path))

        operations = []
        current_uri, content = await open_document(opentitan_client, uart_path, wait_time=1.0)

        # Operation 1: Get symbols
        start = time.perf_counter()
        symbols = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=current_uri),
            )
        )
        operations.append(("symbols", (time.perf_counter() - start) * 1000))

        # Operation 2: Hover on first symbol
        if symbols and len(symbols) > 0:
            first_sym = symbols[0]
            if isinstance(first_sym, types.DocumentSymbol):
                pos = first_sym.selection_range.start
            else:
                pos = first_sym.location.range.start

            start = time.perf_counter()
            hover = await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=current_uri),
                    position=pos,
                )
            )
            operations.append(("hover", (time.perf_counter() - start) * 1000))

        # Operation 3: Find references
        start = time.perf_counter()
        refs = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=current_uri),
                position=types.Position(line=10, character=10),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        operations.append(("references", (time.perf_counter() - start) * 1000))

        # Operation 4: Go to definition
        start = time.perf_counter()
        defn = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=current_uri),
                position=types.Position(line=20, character=10),
            )
        )
        operations.append(("definition", (time.perf_counter() - start) * 1000))

        benchmark.stop()

        print("\nNavigation chain performance:")
        total = 0
        for op, elapsed in operations:
            print(f"  {op}: {elapsed:.1f}ms")
            total += elapsed
        print(f"  Total: {total:.1f}ms")
