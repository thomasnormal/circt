# Iteration 45 - Track D: LSP Rename Support

## Summary

The Verilog LSP server already had full `textDocument/rename` support implemented. This iteration focused on:
1. Verifying the existing implementation
2. Fixing a compilation error in workspace symbol handling
3. Creating comprehensive test coverage

## Implementation Status

### Core Functionality ✓
- **`textDocument/prepareRename`** - Returns the range and current name of the symbol to be renamed
- **`textDocument/rename`** - Returns a `WorkspaceEdit` with all symbol occurrences renamed
- **Server Capabilities** - Correctly advertises `renameProvider: { prepareProvider: true }`

### Supported Symbol Types ✓
The implementation supports renaming:
- Variables (`logic`, `reg`, etc.)
- Nets (`wire`, etc.)
- Ports (input/output/inout)
- Parameters
- Module/class/interface definitions
- Module/class instances
- Subroutines (functions/tasks)

### Validation ✓
The implementation validates:
- Non-empty identifier names
- Valid identifier syntax (starts with letter or underscore)
- Valid characters (alphanumeric, underscore, dollar sign)

### Implementation Details

**Files Modified:**
- `/home/thomas-ahle/circt/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogServer.cpp`
  - Fixed compilation error: Changed `DenseSet<std::string>` to `StringSet<>`
  - Fixed `.str()` call on already-string type

**Files with Existing Implementation:**
- `/home/thomas-ahle/circt/lib/Tools/circt-verilog-lsp-server/LSPServer.cpp` - LSP handlers
- `/home/thomas-ahle/circt/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogDocument.cpp` - Core logic
- `/home/thomas-ahle/circt/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogServer.cpp` - Server coordination
- `/home/thomas-ahle/circt/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogTextFile.cpp` - File handling

## Test Coverage

### Existing Tests
1. **`rename.test`** - Basic rename functionality
   - Signal renaming with multiple references
   - Port renaming
   - `prepareRename` validation

### New Comprehensive Tests Created
2. **`rename-comprehensive.test`** - Advanced scenarios
   - Parameter renaming
   - Function renaming
   - Local variable renaming
   - Complex expressions

3. **`rename-variables.test`** - Multiple reference handling
   - Variable with 5+ references
   - Verifies all occurrences are found
   - Tests declaration + usage sites

4. **`rename-edge-cases.test`** - Validation testing
   - Invalid identifier (starts with number) → Error
   - Empty name → Error
   - Underscore prefix → Success
   - Dollar sign in name → Success

## How Rename Works

### Algorithm
1. **Find Symbol** - Locate symbol at cursor position using interval map
2. **Validate Symbol** - Check if symbol type is renameable
3. **Validate New Name** - Check identifier syntax rules
4. **Find Definition** - Get the symbol's definition location
5. **Find References** - Use reference index to find all usages
6. **Create Edits** - Generate `TextEdit` for definition + all references
7. **Deduplicate** - Sort and remove duplicate edits
8. **Return WorkspaceEdit** - Package all edits for the URI

### Key Features
- Uses existing reference tracking infrastructure
- Handles multiple references correctly
- Deduplicates edits to avoid conflicts
- Validates identifier naming rules
- Works within single file scope (current limitation)

## Test Results

All tests pass successfully:

```bash
cd /home/thomas-ahle/circt/build
./bin/circt-verilog-lsp-server -lit-test < test/rename.test
./bin/circt-verilog-lsp-server -lit-test < test/rename-comprehensive.test
./bin/circt-verilog-lsp-server -lit-test < test/rename-variables.test
./bin/circt-verilog-lsp-server -lit-test < test/rename-edge-cases.test
```

## Example Usage

### Request
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "textDocument/rename",
  "params": {
    "textDocument": {"uri": "file:///test.sv"},
    "position": {"line": 5, "character": 16},
    "newName": "data_reg"
  }
}
```

### Response
```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "changes": {
      "file:///test.sv": [
        {
          "range": {
            "start": {"line": 5, "character": 14},
            "end": {"line": 5, "character": 25}
          },
          "newText": "data_reg"
        },
        {
          "range": {
            "start": {"line": 8, "character": 6},
            "end": {"line": 8, "character": 17}
          },
          "newText": "data_reg"
        }
        // ... more edits for each reference
      ]
    }
  }
}
```

## Known Limitations

1. **Single File Scope** - Currently only renames within the current file. Cross-file renaming would require workspace-wide reference tracking.

2. **No Semantic Validation** - Doesn't check for naming conflicts or shadowing issues with the new name.

3. **No Undo/Redo Tracking** - The LSP server doesn't maintain edit history; clients must handle this.

## Future Enhancements

1. **Cross-file Rename** - Extend to rename symbols across multiple files in a project
2. **Conflict Detection** - Warn if new name conflicts with existing symbols
3. **Preview Changes** - Add support for showing rename preview before applying
4. **Rename Refactoring** - Support for more complex refactorings (e.g., rename module → update all instantiations)

## Conclusion

The Verilog LSP server has robust rename support that:
- ✅ Follows LSP specification
- ✅ Handles multiple references correctly
- ✅ Validates identifier names
- ✅ Supports all major symbol types
- ✅ Has comprehensive test coverage

The implementation leverages the existing reference tracking infrastructure efficiently and provides a solid foundation for IDE rename functionality.
