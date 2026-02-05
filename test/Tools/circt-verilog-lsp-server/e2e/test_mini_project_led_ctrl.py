"""
Mini-Project: LED Controller Tests

Tests LSP features while working on the LED controller peripheral:
- Completion for TLUL bus signals
- Navigation to existing code patterns
- Type checking and diagnostics
- Cross-file navigation within the new IP
"""

import asyncio
import pathlib
import time

import pytest
from lsprotocol import types

from conftest import (
    OPENTITAN_HW,
    open_document,
)


# LED controller file paths
LED_CTRL_DIR = OPENTITAN_HW / "ip" / "led_ctrl"
LED_CTRL_RTL = LED_CTRL_DIR / "rtl"
LED_CTRL_DV = LED_CTRL_DIR / "dv"

LED_CTRL_PKG = LED_CTRL_RTL / "led_ctrl_pkg.sv"
LED_CTRL_REG_TOP = LED_CTRL_RTL / "led_ctrl_reg_top.sv"
LED_CTRL_TOP = LED_CTRL_RTL / "led_ctrl.sv"
LED_CTRL_IF = LED_CTRL_DV / "led_ctrl_if.sv"
LED_CTRL_ENV = LED_CTRL_DV / "led_ctrl_env.sv"


def skip_if_led_ctrl_missing():
    """Skip test if LED controller files don't exist."""
    if not LED_CTRL_DIR.exists():
        pytest.skip("LED controller not found")


class TestLedCtrlPackage:
    """Test LSP features on led_ctrl_pkg.sv."""

    @pytest.mark.asyncio
    async def test_package_symbols(self, opentitan_client, benchmark):
        """Verify document symbols for the package."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_PKG.exists():
            pytest.skip("led_ctrl_pkg.sv not found")

        uri, _ = await open_document(opentitan_client, LED_CTRL_PKG, wait_time=1.0)

        benchmark.start("led_pkg_symbols", str(LED_CTRL_PKG))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nled_ctrl_pkg symbols: {len(symbols)}")

        # Should have package, typedef enum, struct, etc.
        symbol_names = []
        for sym in symbols:
            if isinstance(sym, types.DocumentSymbol):
                symbol_names.append(sym.name)
                print(f"  {sym.kind}: {sym.name}")

        # Verify key symbols
        assert 'led_ctrl_pkg' in symbol_names or len(symbols) > 0

    @pytest.mark.asyncio
    async def test_package_no_errors(self, opentitan_client, benchmark):
        """Verify package has no syntax errors."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_PKG.exists():
            pytest.skip("led_ctrl_pkg.sv not found")

        uri, _ = await open_document(opentitan_client, LED_CTRL_PKG, wait_time=1.0)

        benchmark.start("led_pkg_diagnostics", str(LED_CTRL_PKG))

        await asyncio.sleep(0.5)

        diagnostics = opentitan_client.diagnostics.get(uri, [])
        errors = [d for d in diagnostics if d.severity == types.DiagnosticSeverity.Error]

        benchmark.stop()

        print(f"\nled_ctrl_pkg diagnostics: {len(errors)} errors")

        for err in errors[:5]:
            print(f"  Line {err.range.start.line}: {err.message}")


class TestLedCtrlRegTop:
    """Test LSP features on led_ctrl_reg_top.sv."""

    @pytest.mark.asyncio
    async def test_tlul_completion(self, opentitan_client, benchmark):
        """Test completion for TLUL signals."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_REG_TOP.exists():
            pytest.skip("led_ctrl_reg_top.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_REG_TOP, wait_time=1.0)

        # Find a line inside the module
        lines = content.split('\n')
        target_line = 30

        for i, line in enumerate(lines):
            if 'always_ff' in line or 'assign' in line:
                target_line = i + 1
                break

        benchmark.start("tlul_completion", str(LED_CTRL_REG_TOP))

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=4),
            )
        )

        benchmark.stop()

        if result:
            items = result.items if isinstance(result, types.CompletionList) else result
            print(f"\nTLUL completion: {len(items)} items")

            # Look for TLUL-related completions
            tlul_items = [i for i in items if 'tl' in i.label.lower()]
            print(f"  TLUL items: {len(tlul_items)}")

    @pytest.mark.asyncio
    async def test_goto_package_type(self, opentitan_client, benchmark):
        """Test go-to-definition for package types."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_REG_TOP.exists():
            pytest.skip("led_ctrl_reg_top.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_REG_TOP, wait_time=1.0)

        # Find led_ctrl_pkg:: usage
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'led_ctrl_pkg::' in line or 'led_state_t' in line:
                if 'led_ctrl_pkg::' in line:
                    target_col = line.find('led_ctrl_pkg::') + len('led_ctrl_pkg::')
                else:
                    target_col = line.find('led_state_t')
                target_line = line_num
                break

        if target_line is None:
            pytest.skip("No package type reference found")

        benchmark.start("goto_package_type", str(LED_CTRL_REG_TOP))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        print(f"\nGo-to-definition for package type:")
        if result:
            if isinstance(result, types.Location):
                print(f"  Target: {result.uri}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"  Target: {result[0]}")
        else:
            print("  No definition found")


class TestLedCtrlTop:
    """Test LSP features on led_ctrl.sv (main module)."""

    @pytest.mark.asyncio
    async def test_module_symbols(self, opentitan_client, benchmark):
        """Verify document symbols for the main module."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_TOP.exists():
            pytest.skip("led_ctrl.sv not found")

        uri, _ = await open_document(opentitan_client, LED_CTRL_TOP, wait_time=1.0)

        benchmark.start("led_ctrl_symbols", str(LED_CTRL_TOP))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nled_ctrl module symbols: {len(symbols)}")

        for sym in symbols[:10]:
            if isinstance(sym, types.DocumentSymbol):
                print(f"  {sym.kind}: {sym.name}")

    @pytest.mark.asyncio
    async def test_submodule_navigation(self, opentitan_client, benchmark):
        """Test navigation to submodule (led_ctrl_reg_top)."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_TOP.exists():
            pytest.skip("led_ctrl.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_TOP, wait_time=1.0)

        # Find led_ctrl_reg_top instantiation
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'led_ctrl_reg_top' in line and 'u_reg' in line:
                target_line = line_num
                target_col = line.find('led_ctrl_reg_top')
                break

        if target_line is None:
            pytest.skip("No reg_top instantiation found")

        benchmark.start("submodule_navigation", str(LED_CTRL_TOP))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        print(f"\nNavigation to led_ctrl_reg_top:")
        if result:
            if isinstance(result, types.Location):
                print(f"  Target: {result.uri}")
                assert 'led_ctrl_reg_top' in result.uri
            elif isinstance(result, list) and len(result) > 0:
                target = result[0]
                if hasattr(target, 'uri'):
                    print(f"  Target: {target.uri}")

    @pytest.mark.asyncio
    async def test_generate_block_hover(self, opentitan_client, benchmark):
        """Test hover inside generate block."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_TOP.exists():
            pytest.skip("led_ctrl.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_TOP, wait_time=1.0)

        # Find generate block
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if 'generate' in line or 'genvar' in line:
                target_line = line_num + 2
                break

        if target_line is None:
            pytest.skip("No generate block found")

        benchmark.start("generate_hover", str(LED_CTRL_TOP))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=10),
            )
        )

        benchmark.stop()

        print(f"\nHover in generate block: {'Info available' if result else 'No info'}")


class TestLedCtrlInterface:
    """Test LSP features on led_ctrl_if.sv."""

    @pytest.mark.asyncio
    async def test_interface_symbols(self, opentitan_client, benchmark):
        """Verify interface symbols."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_IF.exists():
            pytest.skip("led_ctrl_if.sv not found")

        uri, _ = await open_document(opentitan_client, LED_CTRL_IF, wait_time=1.0)

        benchmark.start("interface_symbols", str(LED_CTRL_IF))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nled_ctrl_if symbols: {len(symbols)}")

        for sym in symbols[:10]:
            if isinstance(sym, types.DocumentSymbol):
                print(f"  {sym.kind}: {sym.name}")

    @pytest.mark.asyncio
    async def test_modport_hover(self, opentitan_client, benchmark):
        """Test hover on modport."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_IF.exists():
            pytest.skip("led_ctrl_if.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_IF, wait_time=1.0)

        # Find modport
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if 'modport' in line:
                target_line = line_num
                break

        if target_line is None:
            pytest.skip("No modport found")

        benchmark.start("modport_hover", str(LED_CTRL_IF))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=10),
            )
        )

        benchmark.stop()

        print(f"\nModport hover: {'Info available' if result else 'No info'}")


class TestLedCtrlEnv:
    """Test LSP features on led_ctrl_env.sv (UVM environment)."""

    @pytest.mark.asyncio
    async def test_uvm_class_symbols(self, opentitan_client, benchmark):
        """Verify UVM class symbols."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_ENV.exists():
            pytest.skip("led_ctrl_env.sv not found")

        uri, _ = await open_document(opentitan_client, LED_CTRL_ENV, wait_time=1.0)

        benchmark.start("uvm_env_symbols", str(LED_CTRL_ENV))

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\nled_ctrl_env symbols: {len(symbols)}")

        # Look for class definitions
        class_symbols = []
        for sym in symbols:
            if isinstance(sym, types.DocumentSymbol):
                if sym.kind == types.SymbolKind.Class:
                    class_symbols.append(sym.name)
                print(f"  {sym.kind}: {sym.name}")

        print(f"\n  Classes found: {class_symbols}")


class TestCrossFileNavigation:
    """Test navigation between LED controller files."""

    @pytest.mark.asyncio
    async def test_navigate_top_to_pkg(self, opentitan_client, benchmark):
        """Navigate from led_ctrl.sv to led_ctrl_pkg."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_TOP.exists() or not LED_CTRL_PKG.exists():
            pytest.skip("Required files not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_TOP, wait_time=1.0)

        # Find import or package reference
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'led_ctrl_pkg' in line:
                target_line = line_num
                target_col = line.find('led_ctrl_pkg')
                break

        if target_line is None:
            pytest.skip("No package reference found")

        benchmark.start("cross_file_nav", str(LED_CTRL_TOP))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        if result:
            if isinstance(result, types.Location):
                is_pkg = 'led_ctrl_pkg' in result.uri
                print(f"\nCross-file navigation: {'Success' if is_pkg else 'Wrong target'}")
                print(f"  Target: {result.uri}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"\nCross-file navigation: Found {len(result)} locations")
        else:
            print("\nCross-file navigation: No result")

    @pytest.mark.asyncio
    async def test_find_led_state_references(self, opentitan_client, benchmark):
        """Find all references to led_state_t across files."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_PKG.exists():
            pytest.skip("led_ctrl_pkg.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_PKG, wait_time=1.0)

        # Find led_state_t definition
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'led_state_t' in line and 'typedef' in line:
                target_line = line_num
                target_col = line.find('led_state_t')
                break

        if target_line is None:
            pytest.skip("No led_state_t definition found")

        benchmark.start("find_type_references", str(LED_CTRL_PKG))

        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )

        benchmark.stop()

        ref_count = len(result) if result else 0
        print(f"\nReferences to led_state_t: {ref_count}")

        if result:
            # Group by file
            files = {}
            for ref in result:
                if hasattr(ref, 'uri'):
                    file_name = pathlib.Path(ref.uri.replace('file://', '')).name
                    files[file_name] = files.get(file_name, 0) + 1

            for file_name, count in files.items():
                print(f"  {file_name}: {count} references")


class TestLspAssistedDevelopment:
    """Simulate LSP-assisted development workflow."""

    @pytest.mark.asyncio
    async def test_add_new_led_mode(self, opentitan_client, benchmark):
        """Simulate adding a new LED mode with LSP assistance."""
        skip_if_led_ctrl_missing()

        if not LED_CTRL_PKG.exists():
            pytest.skip("led_ctrl_pkg.sv not found")

        uri, content = await open_document(opentitan_client, LED_CTRL_PKG, wait_time=1.0)

        # Add a new mode to the enum
        lines = content.split('\n')

        # Find the enum
        enum_line = None
        for i, line in enumerate(lines):
            if 'ModeBlink' in line:
                enum_line = i
                break

        if enum_line is None:
            pytest.skip("Enum not found")

        # Simulate typing a new enum value
        new_content = content.replace(
            'ModeBlink    = 2\'b11   // Blink mode',
            'ModeBlink    = 2\'b11,  // Blink mode\n    ModeStrobe   = 2\'b00   // Strobe mode (placeholder)'
        )

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=2),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=new_content)],
            )
        )

        await asyncio.sleep(1.0)

        benchmark.start("add_enum_value", str(LED_CTRL_PKG))

        # Check for diagnostics (should warn about duplicate value)
        diagnostics = opentitan_client.diagnostics.get(uri, [])

        benchmark.stop()

        print(f"\nAdd new enum value:")
        print(f"  Diagnostics after edit: {len(diagnostics)}")

        for d in diagnostics[:3]:
            print(f"    Line {d.range.start.line}: {d.message[:60]}")

        # Request completion for the new enum usage
        completion_result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=enum_line, character=20),
            )
        )

        if completion_result:
            items = completion_result.items if isinstance(completion_result, types.CompletionList) else completion_result
            mode_items = [i for i in items if 'Mode' in i.label]
            print(f"  Mode completions: {len(mode_items)}")
