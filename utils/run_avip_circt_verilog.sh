#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_ROOT="${CIRCT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

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

CIRCT_VERILOG="${CIRCT_VERILOG:-$CIRCT_ROOT/build-test/bin/circt-verilog}"
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

def is_ahb_avip(avip_root: pathlib.Path, filelists):
    if avip_root and "ahb" in avip_root.name.lower():
        return True
    for fl in filelists:
        if "ahb" in fl.name.lower():
            return True
        try:
            text = fl.read_text()
        except OSError:
            continue
        if "ahb" in text.lower():
            return True
    return False

def is_i3c_avip(avip_root: pathlib.Path, filelists):
    if avip_root and "i3c" in avip_root.name.lower():
        return True
    for fl in filelists:
        if "i3c" in fl.name.lower():
            return True
        try:
            text = fl.read_text()
        except OSError:
            continue
        if "i3c" in text.lower():
            return True
    return False

def append_after_once(text: str, anchor: str, block: str):
    if anchor not in text:
        return text, False
    return text.replace(anchor, anchor + block, 1), True

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
    if path.name == "SpiMasterMonitorProxy.sv":
        anchor = "    spiMasterMonitorBFM.sampleData(masterPacketStruct, masterConfigStruct);"
        block = """
    if ((masterPacketStruct.noOfMosiBitsTransfer == 0) ||
        (masterPacketStruct.noOfMisoBitsTransfer == 0)) begin
      `uvm_warning("CIRCT_SPI_MONDBG",
                   $sformatf("master zero transfer width mosi_bits=%0d miso_bits=%0d cpol=%0d cpha=%0d",
                             masterPacketStruct.noOfMosiBitsTransfer,
                             masterPacketStruct.noOfMisoBitsTransfer,
                             masterConfigStruct.cpol, masterConfigStruct.cpha))
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
            text = new_text
            changed = True
    if path.name == "SpiSlaveMonitorProxy.sv":
        anchor = "    spiSlaveMonitorBFM.sampleData(slavePacketStruct, slaveConfigStruct);"
        block = """
    if ((slavePacketStruct.noOfMosiBitsTransfer == 0) ||
        (slavePacketStruct.noOfMisoBitsTransfer == 0)) begin
      `uvm_warning("CIRCT_SPI_MONDBG",
                   $sformatf("slave zero transfer width mosi_bits=%0d miso_bits=%0d cpol=%0d cpha=%0d",
                             slavePacketStruct.noOfMosiBitsTransfer,
                             slavePacketStruct.noOfMisoBitsTransfer,
                             slaveConfigStruct.cpol, slaveConfigStruct.cpha))
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
            text = new_text
            changed = True
    return text, changed

def rewrite_ahb_text(path: pathlib.Path, text: str):
    changed = False
    if path.name in ("AhbMasterAgentBFM.sv", "AhbSlaveAgentBFM.sv"):
        bind_starts = (
            "bind AhbMasterMonitorBFM AhbMasterAssertion",
            "bind AhbMasterMonitorBFM AhbMasterCoverProperty",
            "bind AhbSlaveMonitorBFM AhbSlaveAssertion",
            "bind AhbSlaveMonitorBFM AhbSlaveCoverProperty",
        )

        lines = text.splitlines()
        out = []
        in_bind = False
        local_changed = False
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in bind_starts):
                in_bind = True
            if in_bind:
                new_line = line.replace("ahbInterface.", "")
                if new_line != line:
                    local_changed = True
                line = new_line
                if ");" in stripped:
                    in_bind = False
            out.append(line)
        if local_changed:
            text = "\n".join(out) + "\n"
            changed = True

    if path.name == "AhbMasterMonitorProxy.sv":
        anchor = "    ahbMasterMonitorBFM.sampleData (structDataPacket,  structConfigPacket);"
        block = """
    if ((structDataPacket.htrans == 2'b00) || (structDataPacket.hready != 1'b1)) begin
      static int circtAhbMonDbgWarnCount = 0;
      if ((circtAhbMonDbgWarnCount < 8) ||
          ((circtAhbMonDbgWarnCount % 128) == 0)) begin
        `uvm_warning("CIRCT_AHB_MONDBG",
                     $sformatf("master inactive sample htrans=%0d hwrite=%0d hready=%0d haddr=%0h warn_count=%0d",
                               structDataPacket.htrans, structDataPacket.hwrite,
                               structDataPacket.hready, structDataPacket.haddr,
                               circtAhbMonDbgWarnCount))
      end
      circtAhbMonDbgWarnCount++;
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
            text = new_text
            changed = True

    if path.name == "AhbSlaveMonitorProxy.sv":
        anchor = "    ahbSlaveMonitorBFM.slaveSampleData (structDataPacket, structConfigPacket);"
        block = """
    if ((structDataPacket.htrans == 2'b00) || (structDataPacket.hreadyout != 1'b1)) begin
      static int circtAhbMonDbgWarnCount = 0;
      if ((circtAhbMonDbgWarnCount < 8) ||
          ((circtAhbMonDbgWarnCount % 128) == 0)) begin
        `uvm_warning("CIRCT_AHB_MONDBG",
                     $sformatf("slave inactive sample htrans=%0d hwrite=%0d hreadyout=%0d haddr=%0h warn_count=%0d",
                               structDataPacket.htrans, structDataPacket.hwrite,
                               structDataPacket.hreadyout, structDataPacket.haddr,
                               circtAhbMonDbgWarnCount))
      end
      circtAhbMonDbgWarnCount++;
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
            text = new_text
            changed = True
    return text, changed

def rewrite_i3c_text(path: pathlib.Path, text: str):
    changed = False
    if path.name == "i3c_controller_monitor_proxy.sv":
        anchor = "    i3c_controller_mon_bfm_h.sample_data(struct_packet,struct_cfg);"
        block = """
    if ((struct_packet.operation == 1'b0) &&
        (struct_packet.no_of_i3c_bits_transfer == 0)) begin
      `uvm_warning("CIRCT_I3C_MONDBG",
                   $sformatf("controller zero write payload op=%0d addr_ack=%0d bits=%0d target=0x%0h",
                             struct_packet.operation, struct_packet.targetAddressStatus,
                             struct_packet.no_of_i3c_bits_transfer, struct_packet.targetAddress))
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
            text = new_text
            changed = True

    if path.name == "i3c_target_monitor_proxy.sv":
        anchor = "    i3c_target_mon_bfm_h.sample_data(struct_packet,struct_cfg);"
        block = """
    if ((struct_packet.operation == 1'b0) &&
        (struct_packet.no_of_i3c_bits_transfer == 0)) begin
      `uvm_warning("CIRCT_I3C_MONDBG",
                   $sformatf("target zero write payload op=%0d addr_ack=%0d bits=%0d target=0x%0h",
                             struct_packet.operation, struct_packet.targetAddressStatus,
                             struct_packet.no_of_i3c_bits_transfer, struct_packet.targetAddress))
    end
"""
        new_text, did_change = append_after_once(text, anchor, block)
        if did_change:
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
    spi_patch_names = {
        "SpiMasterAssertions.sv",
        "SpiSlaveAssertions.sv",
        "SpiMasterSeqItemConverter.sv",
        "SpiSlaveSeqItemConverter.sv",
        "SpiMasterAgentConfig.sv",
        "SpiSimpleFdRandTest.sv",
        "SpiMasterMonitorProxy.sv",
        "SpiSlaveMonitorProxy.sv",
    }
    spi_passthrough_names = {
        "SpiMasterPkg.sv",
        "SpiSlavePkg.sv",
    }
    file_idx_by_name = {}
    for idx, path_str in enumerate(files):
        name = pathlib.Path(path_str).name
        if name in spi_patch_names and name not in file_idx_by_name:
            file_idx_by_name[name] = idx

    # Copy package files into tmp so local `include` lookup can resolve against
    # patched include-only files emitted in the same directory.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in spi_passthrough_names:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(text)
        files[idx] = str(tmp_path.resolve())

    # Patch filelist entries directly when present.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in spi_patch_names:
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

    # Patch include-only SPI sources and prepend tmp dir in include search path.
    if avip_root and avip_root.exists():
        include_only_patched = False
        for name in spi_patch_names:
            if name in file_idx_by_name:
                continue
            for path in avip_root.rglob(name):
                try:
                    text = path.read_text()
                except OSError:
                    continue
                new_text, changed = rewrite_spi_text(path, text)
                if not changed:
                    continue
                tmp_path = tmp_dir / path.name
                tmp_path.write_text(new_text)
                include_only_patched = True
                break
        if include_only_patched:
            includes.insert(0, str(tmp_dir.resolve()))

needs_ahb = is_ahb_avip(avip_dir, filelists)
if needs_ahb:
    tmp_dir = out_path.parent / ".avip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ahb_patch_names = {
        "AhbMasterAgentBFM.sv",
        "AhbSlaveAgentBFM.sv",
        "AhbMasterMonitorProxy.sv",
        "AhbSlaveMonitorProxy.sv",
    }
    ahb_passthrough_names = {
        "AhbMasterPackage.sv",
        "AhbSlavePackage.sv",
    }
    file_idx_by_name = {}
    for idx, path_str in enumerate(files):
        name = pathlib.Path(path_str).name
        if name in ahb_patch_names and name not in file_idx_by_name:
            file_idx_by_name[name] = idx

    # Copy package files into tmp so local `include` lookup can resolve against
    # patched include-only files emitted in the same directory.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in ahb_passthrough_names:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(text)
        files[idx] = str(tmp_path.resolve())

    # Patch filelist entries directly when present.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in ahb_patch_names:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        new_text, changed = rewrite_ahb_text(path, text)
        if not changed:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(new_text)
        files[idx] = str(tmp_path.resolve())

    # Patch include-only AHB sources and prepend tmp dir in include search path.
    if avip_root and avip_root.exists():
        include_only_patched = False
        for name in ahb_patch_names:
            if name in file_idx_by_name:
                continue
            for path in avip_root.rglob(name):
                try:
                    text = path.read_text()
                except OSError:
                    continue
                new_text, changed = rewrite_ahb_text(path, text)
                if not changed:
                    continue
                tmp_path = tmp_dir / path.name
                tmp_path.write_text(new_text)
                include_only_patched = True
                break
        if include_only_patched:
            includes.insert(0, str(tmp_dir.resolve()))

needs_i3c = is_i3c_avip(avip_dir, filelists)
if needs_i3c:
    tmp_dir = out_path.parent / ".avip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    i3c_patch_names = {
        "i3c_controller_monitor_proxy.sv",
        "i3c_target_monitor_proxy.sv",
    }
    i3c_passthrough_names = {
        "i3c_controller_pkg.sv",
        "i3c_target_pkg.sv",
    }
    file_idx_by_name = {}
    for idx, path_str in enumerate(files):
        name = pathlib.Path(path_str).name
        if name in i3c_patch_names and name not in file_idx_by_name:
            file_idx_by_name[name] = idx

    # Copy package files into tmp so local `include` lookup can resolve against
    # patched include-only files emitted in the same directory.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in i3c_passthrough_names:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(text)
        files[idx] = str(tmp_path.resolve())

    # Patch filelist entries directly when present.
    for idx, path_str in enumerate(list(files)):
        path = pathlib.Path(path_str)
        if path.name not in i3c_patch_names:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        new_text, changed = rewrite_i3c_text(path, text)
        if not changed:
            continue
        tmp_path = tmp_dir / path.name
        tmp_path.write_text(new_text)
        files[idx] = str(tmp_path.resolve())

    # Patch include-only I3C sources and prepend tmp dir in include search path.
    if avip_root and avip_root.exists():
        include_only_patched = False
        for name in i3c_patch_names:
            if name in file_idx_by_name:
                continue
            for path in avip_root.rglob(name):
                try:
                    text = path.read_text()
                except OSError:
                    continue
                new_text, changed = rewrite_i3c_text(path, text)
                if not changed:
                    continue
                tmp_path = tmp_dir / path.name
                tmp_path.write_text(new_text)
                include_only_patched = True
                break
        if include_only_patched:
            includes.insert(0, str(tmp_dir.resolve()))

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
warnings_path = out_path.with_suffix(".warnings.log")
with out_path.open("w") as log, warnings_path.open("w") as wlog:
    result = subprocess.run(cmd, stdout=log, stderr=wlog, text=True)

def sanitize_ir_output(out_path: pathlib.Path, warnings_path: pathlib.Path):
    """Ensure OUT starts at the first MLIR module line.

    Some front-end diagnostics are emitted on stdout and can otherwise pollute
    the IR file, causing circt-sim parse failures.
    """
    try:
        text = out_path.read_text()
    except OSError:
        return
    lines = text.splitlines(keepends=True)
    if not lines:
        return

    def is_module_start(line: str):
        stripped = line.lstrip()
        return (stripped.startswith("module ") or
                stripped.startswith("module{") or
                stripped.startswith("module\t") or
                stripped.startswith("module\n") or
                stripped.startswith("module{") or
                stripped.startswith("module {") or
                stripped.startswith("builtin.module"))

    start = None
    for i, line in enumerate(lines):
        if is_module_start(line):
            start = i
            break
    if start is None or start == 0:
        return

    try:
        with warnings_path.open("a") as wlog:
            wlog.write("\n# stdout diagnostics moved from IR output\n")
            for line in lines[:start]:
                wlog.write(line)
        with out_path.open("w") as out:
            for line in lines[start:]:
                out.write(line)
    except OSError:
        return

if result.returncode == 0:
    sanitize_ir_output(out_path, warnings_path)

print(result.returncode)
print(str(out_path))
raise SystemExit(result.returncode)
PY
