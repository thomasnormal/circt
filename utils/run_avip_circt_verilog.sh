#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <avip_dir> [filelist...]" >&2
  exit 1
fi

AVIP_DIR="$1"
shift || true

FILELISTS_ARRAY=()
if [[ $# -gt 0 ]]; then
  FILELISTS_ARRAY=("$@")
fi
if [[ ${#FILELISTS_ARRAY[@]} -eq 0 ]]; then
  if [[ -n "${FILELIST:-}" ]]; then
    FILELISTS_ARRAY=("$FILELIST")
  else
    mapfile -t candidates < <(find "$AVIP_DIR/sim" -maxdepth 2 -iname "*compile*.f" 2>/dev/null || true)
    if [[ ${#candidates[@]} -eq 0 ]]; then
      echo "no filelist found under $AVIP_DIR/sim (set FILELIST or pass as args)" >&2
      exit 1
    fi
    FILELISTS_ARRAY=("${candidates[0]}")
  fi
fi

for filelist in "${FILELISTS_ARRAY[@]}"; do
  if [[ ! -f "$filelist" ]]; then
    echo "filelist not found: $filelist" >&2
    exit 1
  fi
done

FILELIST_BASE="${FILELIST_BASE:-}"
if [[ -n "$FILELIST_BASE" && ! -d "$FILELIST_BASE" ]]; then
  echo "FILELIST_BASE does not exist: $FILELIST_BASE" >&2
  exit 1
fi

CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
UVM_DIR="${UVM_DIR:-/home/thomas-ahle/uvm-core/src}"
OUT="${OUT:-$PWD/avip-circt-verilog.log}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
if [[ -z "${TIMESCALE+x}" ]]; then
  TIMESCALE="1ns/1ps"
fi
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_VERILOG_IR="${CIRCT_VERILOG_IR:-moore}"
FILELISTS_STR="$(printf "%s\n" "${FILELISTS_ARRAY[@]}")"
export FILELISTS_STR

python3 - <<'PY' \
  "$FILELIST_BASE" "$CIRCT_VERILOG" "$UVM_DIR" "$OUT" \
  "$DISABLE_UVM_AUTO_INCLUDE" "$TIMESCALE" "$CIRCT_VERILOG_ARGS" \
  "$CIRCT_VERILOG_IR" "$AVIP_DIR"
import os
import pathlib
import re
import subprocess
import sys
import shlex

base_override = pathlib.Path(sys.argv[1]) if sys.argv[1] else None
circt_verilog = sys.argv[2]
uvm_dir = pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])
disable_auto = sys.argv[5] != "0"
timescale = sys.argv[6]
extra_args = shlex.split(sys.argv[7]) if sys.argv[7] else []
ir_mode = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] else "moore"
avip_dir = pathlib.Path(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] else None
filelists = [pathlib.Path(p) for p in os.environ.get("FILELISTS_STR", "").split("\n") if p]

if not filelists:
    raise SystemExit("no filelists provided")

avip_root = avip_dir.resolve() if avip_dir else None

def ensure_axi4lite_env(avip_root: pathlib.Path, filelists):
    if not avip_root or not avip_root.exists():
        return False
    avip_root = avip_root.resolve()
    needs_axi4lite = False
    for fl in filelists:
        try:
            text = fl.read_text()
        except OSError:
            continue
        if "AXI4LITE_" in text or "axi4lite" in text.lower():
            needs_axi4lite = True
            break
    if not needs_axi4lite:
        if "axi4lite" in avip_root.name.lower():
            needs_axi4lite = True
        else:
            for fl in filelists:
                if "axi4lite" in fl.name.lower():
                    needs_axi4lite = True
                    break
    if not needs_axi4lite:
        return False
    env = os.environ
    env.setdefault("AXI4LITE_PROJECT", str(avip_root))
    master = avip_root / "src/axi4LiteMasterVIP"
    slave = avip_root / "src/axi4LiteSlaveVIP"
    if "AXI4LITE_MASTER" not in env and master.exists():
        env["AXI4LITE_MASTER"] = str(master)
    if "AXI4LITE_SLAVE" not in env and slave.exists():
        env["AXI4LITE_SLAVE"] = str(slave)
    master_write = master / "src/axi4LiteMasterWriteVIP"
    master_read = master / "src/axi4LiteMasterReadVIP"
    slave_write = slave / "src/axi4LiteSlaveWriteVIP"
    slave_read = slave / "src/axi4LiteSlaveReadVIP"
    if "AXI4LITE_MASTERWRITE" not in env and master_write.exists():
        env["AXI4LITE_MASTERWRITE"] = str(master_write)
    if "AXI4LITE_MASTERREAD" not in env and master_read.exists():
        env["AXI4LITE_MASTERREAD"] = str(master_read)
    if "AXI4LITE_SLAVEWRITE" not in env and slave_write.exists():
        env["AXI4LITE_SLAVEWRITE"] = str(slave_write)
    if "AXI4LITE_SLAVEREAD" not in env and slave_read.exists():
        env["AXI4LITE_SLAVEREAD"] = str(slave_read)
    return True

def maybe_prepend_axi4lite_filelists(filelists):
    env_map = {
        "AXI4LITE_MASTERWRITE": "Axi4LiteWriteMaster.f",
        "AXI4LITE_MASTERREAD": "Axi4LiteReadMaster.f",
        "AXI4LITE_SLAVEWRITE": "Axi4LiteWriteSlave.f",
        "AXI4LITE_SLAVEREAD": "Axi4LiteReadSlave.f",
    }
    existing = {fl.resolve() for fl in filelists}
    added = False
    prepended = []
    for env_name, file_name in env_map.items():
        root = os.environ.get(env_name)
        if not root:
            continue
        candidate = pathlib.Path(root) / "sim" / file_name
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in existing:
                prepended.append(resolved)
                existing.add(resolved)
                added = True
    if added:
        # Ensure VIP packages are compiled before env packages.
        filelists[:] = prepended + [fl for fl in filelists if fl.resolve() not in prepended]
    return added

needs_axi4lite = ensure_axi4lite_env(avip_dir, filelists)
if needs_axi4lite:
    maybe_prepend_axi4lite_filelists(filelists)

def is_spi_avip(avip_root: pathlib.Path, filelists):
    if avip_root and "spi" in avip_root.name.lower():
        return True
    for fl in filelists:
        if "spi" in fl.name.lower():
            return True
        try:
            text = fl.read_text()
        except OSError:
            continue
        if "spi_avip" in text.lower():
            return True
    return False

def is_jtag_avip(avip_root: pathlib.Path, filelists):
    if avip_root and "jtag" in avip_root.name.lower():
        return True
    for fl in filelists:
        if "jtag" in fl.name.lower():
            return True
        try:
            text = fl.read_text()
        except OSError:
            continue
        if "jtag" in text.lower():
            return True
    return False

def drop_nested_block_comments(text: str):
    lines = text.splitlines()
    out = []
    in_block = False
    changed = False
    for line in lines:
        if "/*" in line:
            if in_block:
                line = line.replace("/*", "//", 1)
                changed = True
            else:
                in_block = True
        if "*/" in line and in_block:
            in_block = False
        out.append(line)
    return "\n".join(out) + "\n", changed

def rewrite_spi_text(path: pathlib.Path, text: str):
    changed = False
    if path.name in ("SpiMasterAssertions.sv", "SpiSlaveAssertions.sv"):
        text, nested_changed = drop_nested_block_comments(text)
        changed |= nested_changed
    if path.name in ("SpiMasterSeqItemConverter.sv", "SpiSlaveSeqItemConverter.sv"):
        new_text = text.replace("masterInSlaveOut[%0d]\",i,)", "masterInSlaveOut[%0d]\",i)")
        if new_text != text:
            text = new_text
            changed = True
    if path.name == "SpiMasterAgentConfig.sv":
        new_text = text.replace("[11:$]", "[11:1023]")
        if new_text != text:
            text = new_text
            changed = True
    if path.name == "SpiSimpleFdRandTest.sv":
        new_text = re.sub(r"randomize\\(\\)\\s*with\\s*\\{[^}]*\\}",
                          "randomize()", text, flags=re.S)
        if new_text != text:
            text = new_text
            changed = True
    return text, changed

def rewrite_jtag_text(path: pathlib.Path, text: str):
    changed = False
    if path.name == "JtagControllerDeviceAgentBfm.sv":
        new_text = text.replace(
            "bind jtagControllerDeviceMonitorBfm ",
            "bind JtagControllerDeviceMonitorBfm ",
        )
        if new_text != text:
            text = new_text
            changed = True
    if path.name == "JtagTargetDeviceDriverBfm.sv":
        new_text = text.replace(
            "registerBank[instructionRegister]",
            "registerBank[JtagInstructionOpcodeEnum'(instructionRegister)]",
        )
        if new_text != text:
            text = new_text
            changed = True
    return text, changed

def resolve_filelist_path(raw: str, base: pathlib.Path) -> pathlib.Path:
    expanded = pathlib.Path(os.path.expandvars(raw))
    if expanded.is_absolute():
        return expanded.resolve()
    candidate = (base / expanded).resolve()
    if candidate.exists():
        return candidate
    if avip_root and avip_root.exists():
        parts = expanded.parts
        if len(parts) >= 3 and parts[0] == ".." and parts[1] == ".." and parts[2] == "src":
            alt = (avip_root / pathlib.Path(*parts[2:])).resolve()
            if alt.exists():
                return alt
    return candidate

includes = []
files = []
for filelist in filelists:
    base = base_override or filelist.parent
    for line in filelist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        line = re.sub(r"\$\(([^)]+)\)", r"${\1}", line)
        if line.startswith("+incdir+"):
            raw = line[len("+incdir+"):]
            inc = resolve_filelist_path(raw, base)
            # Skip file paths that are mistakenly used as +incdir+
            if inc.is_dir():
                includes.append(str(inc))
            elif inc.is_file():
                # If it's a file, use parent directory instead
                includes.append(str(inc.parent))
        else:
            files.append(str(resolve_filelist_path(line, base)))

# Work around slang bind limitation: Axi4LiteHdlTop includes
# Axi4LiteCoverProperty.sv locally, and slang fails to resolve the bind target
# when the include is present. Rewrite Axi4LiteHdlTop to drop that include and
# rely on the separately compiled interface definition instead.
if needs_axi4lite:
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name != "Axi4LiteHdlTop.sv":
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        lines = text.splitlines()
        changed = False
        for i, line in enumerate(lines):
            if "Axi4LiteCoverProperty.sv" in line and "`include" in line:
                lines[i] = "// circt: dropped Axi4LiteCoverProperty include for bind resolution"
                changed = True
        if not changed:
            continue
        tmp_dir = out_path.parent / ".avip_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / path.name
        tmp_path.write_text("\n".join(lines) + "\n")
        files[idx] = str(tmp_path.resolve())

needs_spi = is_spi_avip(avip_dir, filelists)
if needs_spi:
    tmp_dir = out_path.parent / ".avip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in (
            "SpiMasterAssertions.sv",
            "SpiSlaveAssertions.sv",
            "SpiMasterSeqItemConverter.sv",
            "SpiSlaveSeqItemConverter.sv",
            "SpiMasterAgentConfig.sv",
            "SpiSimpleFdRandTest.sv",
        ):
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        new_text, changed = rewrite_spi_text(path, text)
        if not changed:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(new_text)
        files[idx] = str(tmp_path.resolve())

needs_jtag = is_jtag_avip(avip_dir, filelists)
if needs_jtag:
    tmp_dir = out_path.parent / ".avip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in (
            "JtagControllerDeviceAgentBfm.sv",
            "JtagTargetDeviceDriverBfm.sv",
        ):
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        new_text, changed = rewrite_jtag_text(path, text)
        if not changed:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(new_text)
        files[idx] = str(tmp_path.resolve())

cmd = [circt_verilog, f"--ir-{ir_mode}"]
if disable_auto:
    cmd.append("--no-uvm-auto-include")
if timescale:
    cmd.append(f"--timescale={timescale}")
cmd += ["-I", str(uvm_dir), str(uvm_dir / "uvm_pkg.sv")]
for inc in includes:
    cmd += ["-I", inc]
cmd += extra_args
cmd += files

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

print(result.returncode)
print(str(out_path))
raise SystemExit(result.returncode)
PY
