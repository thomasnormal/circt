# -*- Python -*-

import os
import platform
import re
import shutil
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'CIRCT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    '.aag', '.td', '.mlir', '.lib', '.ll', '.fir', '.sv', '.test'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.mlir_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.circt_tools_dir, append_path=True)

tool_dirs = [
    config.circt_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir
]
tools = [
    'arcilator', 'circt-as', 'circt-bmc', 'circt-capi-synth-test',
    'circt-capi-ir-test', 'circt-capi-om-test', 'circt-capi-firrtl-test',
    'circt-capi-firtool-test', 'circt-capi-rtg-test', 'circt-capi-rtgtest-test',
    'circt-capi-support-test', 'circt-cov', 'circt-dis', 'circt-lec',
    'circt-mut', 'circt-sim', 'circt-sim-compile',
    'circt-reduce', 'circt-synth', 'circt-test', 'circt-translate',
    'domaintool', 'firld', 'firtool', 'hlstool', 'om-linker', 'kanagawatool'
]

if "CIRCT_OPT_CHECK_IR_ROUNDTRIP" in os.environ:
  tools.extend([
      ToolSubst("circt-opt", "circt-opt --verify-roundtrip",
                unresolved="fatal"),
  ])
else:
  tools.extend(["circt-opt"])

# Enable Verilator if it has been detected.
if config.verilator_path != "":
  tool_dirs.append(os.path.dirname(config.verilator_path))
  tools.append('verilator')
  config.available_features.add('verilator')

if config.z3_path:
  llvm_config.with_environment('PATH', config.z3_path, append_path=True)
  config.available_features.add('z3')
  config.substitutions.append(('%z3', os.path.join(config.z3_path, 'z3')))

if config.zlib == "1":
  config.available_features.add('zlib')

# Enable tests for schedulers relying on an external solver from OR-Tools.
if config.scheduling_or_tools != "":
  config.available_features.add('or-tools')

# Add circt-verilog if the Slang frontend is enabled. Some local builds expose
# circt-verilog even when the generated lit site config doesn't set
# `slang_frontend_enabled`; detect that binary as a fallback so SVA tests run.
slang_frontend_enabled = bool(config.slang_frontend_enabled)
if not slang_frontend_enabled:
  search_path = os.pathsep.join(tool_dirs + [config.environment.get('PATH', '')])
  slang_frontend_enabled = shutil.which('circt-verilog', path=search_path) is not None

if slang_frontend_enabled:
  config.available_features.add('slang')
  if 'circt-verilog' not in tools:
    tools.append('circt-verilog')
  if 'circt-verilog-lsp-server' not in tools:
    tools.append('circt-verilog-lsp-server')

# Expose tool features only when those binaries are available. This keeps
# REQUIRES constraints accurate for mixed configurations.
tool_search_path = os.pathsep.join(tool_dirs + [config.environment.get('PATH', '')])
if shutil.which('circt-sim', path=tool_search_path):
  config.available_features.add('circt-sim')

# Enable UVM-gated tests when a usable UVM runtime source tree is available.
repo_root = os.path.normpath(os.path.join(config.test_source_root, '..'))
uvm_candidates = [
    os.path.join(repo_root, 'lib', 'Runtime', 'uvm-core', 'src'),
    os.path.join(repo_root, 'lib', 'Runtime', 'uvm'),
    config.environment.get('CIRCT_UVM_PATH', ''),
    config.environment.get('UVM_PATH', ''),
    config.environment.get('UVM_HOME', ''),
]
if any(path and os.path.isdir(path) for path in uvm_candidates):
  config.available_features.add('uvm')

llvm_config.add_tool_substitutions(tools, tool_dirs)
