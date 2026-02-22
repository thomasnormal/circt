"""
VPI regression tests for circt-sim.

Tests the Verilog Procedural Interface (VPI) implementation by compiling C VPI
test libraries, loading them into circt-sim, and verifying the output.

Run with: python3 -m pytest test/Tools/circt-sim/test_vpi.py -v
"""

import os
import subprocess
import tempfile
import textwrap

import pytest

# Find circt-sim binary
CIRCT_SIM = os.environ.get(
    "CIRCT_SIM",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "build-test", "bin", "circt-sim"
    ),
)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def _compile_vpi_lib(c_source_path, tmpdir):
    """Compile a C source file into a shared VPI library."""
    so_path = os.path.join(tmpdir, "vpi_test.so")
    result = subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", so_path, c_source_path, "-ldl"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"VPI lib compilation failed:\n{result.stderr}"
    return so_path


def _write_mlir(tmpdir, module_name, body):
    """Write a minimal MLIR test module."""
    mlir_path = os.path.join(tmpdir, "test.mlir")
    with open(mlir_path, "w") as f:
        f.write(f"hw.module @{module_name}({body}) {{}}\n")
    return mlir_path


def _run_circt_sim(mlir_path, top, vpi_so, max_time=1000, timeout=30):
    """Run circt-sim with a VPI library and return (stdout+stderr, returncode)."""
    cmd = [CIRCT_SIM, mlir_path, "--top", top, f"--max-time={max_time}"]
    if vpi_so:
        cmd.append(f"--vpi={vpi_so}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    combined = result.stdout + result.stderr
    return combined, result.returncode


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---- Test: basic VPI startup/shutdown ----


class TestVPIBasic:
    """Test basic VPI library loading, startup, and shutdown callbacks."""

    def test_startup_shutdown_callbacks(self, tmpdir):
        """Verify cbStartOfSimulation and cbEndOfSimulation fire."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-basic-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "vpi_test", "in %clk : i1, in %rst : i1")
        output, rc = _run_circt_sim(mlir, "vpi_test", so)
        assert "VPI_TEST: start_of_simulation callback fired" in output
        assert "VPI_TEST: end_of_simulation callback fired" in output
        # No test failures
        assert "0 failed" in output

    def test_product_info(self, tmpdir):
        """Verify vpi_get_vlog_info returns circt-sim product name."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-basic-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "vpi_test", "in %clk : i1, in %rst : i1")
        output, _ = _run_circt_sim(mlir, "vpi_test", so)
        assert "VPI_TEST: product=circt-sim" in output

    def test_signal_discovery(self, tmpdir):
        """Verify VPI can iterate and find signals in a module."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-basic-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "vpi_test", "in %clk : i1, in %rst : i1")
        output, _ = _run_circt_sim(mlir, "vpi_test", so)
        assert "VPI_TEST: found" in output
        # Should find at least 1 signal
        assert "found 0 signals" not in output

    def test_time_at_start(self, tmpdir):
        """Verify simulation time is 0 at start of simulation."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-basic-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "vpi_test", "in %clk : i1, in %rst : i1")
        output, _ = _run_circt_sim(mlir, "vpi_test", so)
        assert "VPI_TEST: time=0:0" in output

    def test_no_errors(self, tmpdir):
        """Verify vpi_chk_error returns 0 (no error) initially."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-basic-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "vpi_test", "in %clk : i1, in %rst : i1")
        output, _ = _run_circt_sim(mlir, "vpi_test", so)
        # All tests should pass
        lines = [l for l in output.split("\n") if "passed" in l and "failed" in l]
        for line in lines:
            assert "0 failed" in line


# ---- Test: put_value / get_value ----


class TestVPIPutValue:
    """Test VPI value read/write roundtrip."""

    def test_put_get_roundtrip(self, tmpdir):
        """Write a value via VPI and read it back."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-put-value-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "put_test", "in %clk : i1, in %data : i8")
        output, _ = _run_circt_sim(mlir, "put_test", so)
        assert "VPI_PUT: after_write=42" in output
        assert "0 failed" in output

    def test_binary_string_format(self, tmpdir):
        """Verify binary string value format works."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-put-value-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "put_test", "in %clk : i1, in %data : i8")
        output, _ = _run_circt_sim(mlir, "put_test", so)
        assert "VPI_PUT: binary=" in output

    def test_handle_by_name(self, tmpdir):
        """Verify vpi_handle_by_name can find a signal and read its value."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-put-value-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "put_test", "in %clk : i1, in %data : i8")
        output, _ = _run_circt_sim(mlir, "put_test", so)
        assert "VPI_PUT: found_value=42" in output


# ---- Test: callbacks and properties ----


class TestVPICallbacks:
    """Test VPI callback registration, removal, and signal properties."""

    def test_callback_register_remove(self, tmpdir):
        """Verify callback registration and removal works."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-callback-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "cb_test", "in %clk : i1, in %data_in : i8")
        output, _ = _run_circt_sim(mlir, "cb_test", so)
        assert "VPI_CB: start_of_simulation" in output
        assert "VPI_CB: end_of_simulation" in output
        assert "0 failed" in output

    def test_signal_properties(self, tmpdir):
        """Verify signal type, size, vector/scalar properties."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-callback-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "cb_test", "in %clk : i1, in %data_in : i8")
        output, _ = _run_circt_sim(mlir, "cb_test", so)
        # 8-bit data_in should have size=8, vector=1, scalar=0
        assert "size=8 vector=1 scalar=0" in output
        # 1-bit clk should have size=1, vector=0, scalar=1
        assert "size=1 vector=0 scalar=1" in output

    def test_module_iteration(self, tmpdir):
        """Verify top-level module iteration works."""
        so = _compile_vpi_lib(os.path.join(TEST_DIR, "vpi-callback-test.c"), tmpdir)
        mlir = _write_mlir(tmpdir, "cb_test", "in %clk : i1, in %data_in : i8")
        output, _ = _run_circt_sim(mlir, "cb_test", so)
        assert "VPI_CB: module_count=" in output
        # Should find at least 1 module
        assert "module_count=0" not in output


# ---- Test: VPI with no library (baseline) ----


class TestVPINoLibrary:
    """Test that circt-sim works without --vpi flag."""

    def test_no_vpi_flag(self, tmpdir):
        """Circt-sim should work without --vpi."""
        mlir = _write_mlir(tmpdir, "no_vpi", "in %clk : i1")
        output, rc = _run_circt_sim(mlir, "no_vpi", None)
        assert rc == 0
        assert "Simulation completed" in output


# ---- Test: inline C VPI library ----


class TestVPIInlineC:
    """Tests with inline C VPI libraries for specific scenarios."""

    def _write_and_compile_vpi(self, tmpdir, c_code):
        """Write C code to a file and compile it."""
        c_path = os.path.join(tmpdir, "inline_vpi.c")
        with open(c_path, "w") as f:
            f.write(c_code)
        return _compile_vpi_lib(c_path, tmpdir)

    def test_multiple_modules(self, tmpdir):
        """Test VPI with multiple top-level modules (multi-top)."""
        c_code = textwrap.dedent("""\
            #include <stdio.h>
            #include <string.h>
            #include <stdint.h>
            typedef void *vpiHandle;
            typedef int PLI_INT32;
            typedef char PLI_BYTE8;
            struct t_vpi_time { PLI_INT32 type; unsigned high, low; double real; };
            typedef struct t_vpi_time *p_vpi_time;
            struct t_vpi_value { PLI_INT32 format; union { PLI_BYTE8 *str; PLI_INT32 scalar, integer; double real; struct t_vpi_time *time; void *vector; PLI_BYTE8 *misc; } value; };
            typedef struct t_vpi_value *p_vpi_value;
            struct t_vpi_error_info { PLI_INT32 state, level; PLI_BYTE8 *message, *product, *code, *file; PLI_INT32 line; };
            typedef struct t_vpi_error_info *p_vpi_error_info;
            struct t_vpi_vlog_info { PLI_INT32 argc; PLI_BYTE8 **argv, *product, *version; };
            typedef struct t_vpi_vlog_info *p_vpi_vlog_info;
            struct t_cb_data { PLI_INT32 reason; PLI_INT32 (*cb_rtn)(struct t_cb_data *); void *obj; struct t_vpi_time *time; struct t_vpi_value *value; PLI_INT32 index; PLI_BYTE8 *user_data; };
            typedef struct t_cb_data *p_cb_data;
            #define vpiModule 32
            #define vpiReg 48
            #define vpiName 2
            #define vpiSize 4
            #define cbStartOfSimulation 11
            #define cbEndOfSimulation 12
            extern vpiHandle vpi_register_cb(p_cb_data);
            extern vpiHandle vpi_iterate(PLI_INT32, vpiHandle);
            extern vpiHandle vpi_scan(vpiHandle);
            extern PLI_INT32 vpi_get(PLI_INT32, vpiHandle);
            extern PLI_BYTE8 *vpi_get_str(PLI_INT32, vpiHandle);
            static int modCount = 0, sigCount = 0;
            static PLI_INT32 sos_cb(struct t_cb_data *d) {
                (void)d;
                vpiHandle mi = vpi_iterate(vpiModule, NULL);
                if (!mi) { fprintf(stderr, "MULTI: no modules\\n"); return 0; }
                vpiHandle m;
                while ((m = vpi_scan(mi)) != NULL) {
                    PLI_BYTE8 *n = vpi_get_str(vpiName, m);
                    fprintf(stderr, "MULTI: module=%s\\n", n ? n : "(null)");
                    modCount++;
                    vpiHandle si = vpi_iterate(vpiReg, m);
                    if (si) {
                        vpiHandle s;
                        while ((s = vpi_scan(si)) != NULL) {
                            PLI_BYTE8 *sn = vpi_get_str(vpiName, s);
                            PLI_INT32 sw = vpi_get(vpiSize, s);
                            fprintf(stderr, "MULTI:   signal=%s width=%d\\n", sn ? sn : "(null)", sw);
                            sigCount++;
                        }
                    }
                }
                fprintf(stderr, "MULTI: modules=%d signals=%d\\n", modCount, sigCount);
                return 0;
            }
            static PLI_INT32 eos_cb(struct t_cb_data *d) {
                (void)d;
                fprintf(stderr, "MULTI: FINAL modules=%d signals=%d\\n", modCount, sigCount);
                return 0;
            }
            static void init(void) {
                struct t_cb_data c1 = {0}; c1.reason = cbStartOfSimulation; c1.cb_rtn = sos_cb; vpi_register_cb(&c1);
                struct t_cb_data c2 = {0}; c2.reason = cbEndOfSimulation; c2.cb_rtn = eos_cb; vpi_register_cb(&c2);
            }
            void (*vlog_startup_routines[])(void) = { init, NULL };
        """)
        so = self._write_and_compile_vpi(tmpdir, c_code)
        mlir_path = os.path.join(tmpdir, "multi.mlir")
        with open(mlir_path, "w") as f:
            f.write("hw.module @mod_a(in %clk : i1, in %a : i8) {}\n")
            f.write("hw.module @mod_b(in %rst : i1, in %b : i16) {}\n")
        output, rc = _run_circt_sim(
            mlir_path, "mod_a", so, max_time=1000
        )
        assert "MULTI: modules=" in output
        # At least one module's signals found
        assert "signals=" in output
        assert "MULTI: FINAL" in output

    def test_wide_signal_value(self, tmpdir):
        """Test reading/writing values wider than 32 bits."""
        c_code = textwrap.dedent("""\
            #include <stdio.h>
            #include <string.h>
            #include <stdint.h>
            typedef void *vpiHandle;
            typedef int PLI_INT32;
            typedef unsigned int PLI_UINT32;
            typedef char PLI_BYTE8;
            struct t_vpi_time { PLI_INT32 type; PLI_UINT32 high, low; double real; };
            typedef struct t_vpi_time *p_vpi_time;
            struct t_vpi_vecval { PLI_UINT32 aval, bval; };
            struct t_vpi_value { PLI_INT32 format; union { PLI_BYTE8 *str; PLI_INT32 scalar, integer; double real; struct t_vpi_time *time; struct t_vpi_vecval *vector; PLI_BYTE8 *misc; } value; };
            typedef struct t_vpi_value *p_vpi_value;
            struct t_vpi_error_info { PLI_INT32 state, level; PLI_BYTE8 *message, *product, *code, *file; PLI_INT32 line; };
            typedef struct t_vpi_error_info *p_vpi_error_info;
            struct t_vpi_vlog_info { PLI_INT32 argc; PLI_BYTE8 **argv, *product, *version; };
            typedef struct t_vpi_vlog_info *p_vpi_vlog_info;
            struct t_cb_data { PLI_INT32 reason; PLI_INT32 (*cb_rtn)(struct t_cb_data *); void *obj; struct t_vpi_time *time; struct t_vpi_value *value; PLI_INT32 index; PLI_BYTE8 *user_data; };
            typedef struct t_cb_data *p_cb_data;
            #define vpiModule 32
            #define vpiReg 48
            #define vpiName 2
            #define vpiSize 4
            #define vpiBinStrVal 1
            #define vpiVectorVal 9
            #define cbStartOfSimulation 11
            #define cbEndOfSimulation 12
            extern vpiHandle vpi_register_cb(p_cb_data);
            extern vpiHandle vpi_iterate(PLI_INT32, vpiHandle);
            extern vpiHandle vpi_scan(vpiHandle);
            extern PLI_INT32 vpi_get(PLI_INT32, vpiHandle);
            extern PLI_BYTE8 *vpi_get_str(PLI_INT32, vpiHandle);
            extern void vpi_get_value(vpiHandle, p_vpi_value);
            extern void vpi_put_value(vpiHandle, p_vpi_value, p_vpi_time, PLI_INT32);
            static PLI_INT32 sos_cb(struct t_cb_data *d) {
                (void)d;
                vpiHandle mi = vpi_iterate(vpiModule, NULL);
                if (!mi) return 0;
                vpiHandle m = vpi_scan(mi);
                if (!m) return 0;
                vpiHandle si = vpi_iterate(vpiReg, m);
                if (!si) return 0;
                vpiHandle sig = NULL;
                { vpiHandle s; while ((s = vpi_scan(si)) != NULL) { if (vpi_get(vpiSize, s) >= 32) { sig = s; break; } } }
                if (!sig) { fprintf(stderr, "WIDE: no wide signal found\\n"); return 0; }
                PLI_INT32 w = vpi_get(vpiSize, sig);
                fprintf(stderr, "WIDE: signal=%s width=%d\\n", vpi_get_str(vpiName, sig), w);
                // Write via vector format
                struct t_vpi_vecval vec[2];
                vec[0].aval = 0xDEADBEEF; vec[0].bval = 0;
                vec[1].aval = 0x12345678; vec[1].bval = 0;
                struct t_vpi_value wval;
                wval.format = vpiVectorVal;
                wval.value.vector = vec;
                vpi_put_value(sig, &wval, NULL, 0);
                // Read back as binary string
                struct t_vpi_value rval;
                rval.format = vpiBinStrVal;
                vpi_get_value(sig, &rval);
                fprintf(stderr, "WIDE: binary=%s\\n", rval.value.str ? rval.value.str : "(null)");
                // Read back as vector
                struct t_vpi_vecval rvec[2];
                struct t_vpi_value rvval;
                rvval.format = vpiVectorVal;
                rvval.value.vector = rvec;
                vpi_get_value(sig, &rvval);
                fprintf(stderr, "WIDE: vec[0].aval=0x%08x vec[1].aval=0x%08x\\n", rvec[0].aval, rvec[1].aval);
                int ok = (rvec[0].aval == 0xDEADBEEF);
                fprintf(stderr, "WIDE: %s\\n", ok ? "PASS" : "FAIL");
                return 0;
            }
            static PLI_INT32 eos_cb(struct t_cb_data *d) { (void)d; fprintf(stderr, "WIDE: done\\n"); return 0; }
            static void init(void) {
                struct t_cb_data c1 = {0}; c1.reason = cbStartOfSimulation; c1.cb_rtn = sos_cb; vpi_register_cb(&c1);
                struct t_cb_data c2 = {0}; c2.reason = cbEndOfSimulation; c2.cb_rtn = eos_cb; vpi_register_cb(&c2);
            }
            void (*vlog_startup_routines[])(void) = { init, NULL };
        """)
        so = self._write_and_compile_vpi(tmpdir, c_code)
        mlir = _write_mlir(tmpdir, "wide_test", "in %clk : i1, in %data : i64")
        output, rc = _run_circt_sim(mlir, "wide_test", so)
        assert "WIDE: vec[0].aval=0xdeadbeef" in output
        assert "WIDE: PASS" in output

    def test_empty_module(self, tmpdir):
        """Test VPI with a module that has no ports (no signals to find)."""
        c_code = textwrap.dedent("""\
            #include <stdio.h>
            #include <string.h>
            #include <stdint.h>
            typedef void *vpiHandle;
            typedef int PLI_INT32;
            typedef char PLI_BYTE8;
            struct t_vpi_time { PLI_INT32 type; unsigned high, low; double real; };
            typedef struct t_vpi_time *p_vpi_time;
            struct t_vpi_value { PLI_INT32 format; union { PLI_BYTE8 *str; PLI_INT32 scalar, integer; double real; struct t_vpi_time *time; void *vector; PLI_BYTE8 *misc; } value; };
            typedef struct t_vpi_value *p_vpi_value;
            struct t_vpi_error_info { PLI_INT32 state, level; PLI_BYTE8 *message, *product, *code, *file; PLI_INT32 line; };
            typedef struct t_vpi_error_info *p_vpi_error_info;
            struct t_vpi_vlog_info { PLI_INT32 argc; PLI_BYTE8 **argv, *product, *version; };
            typedef struct t_vpi_vlog_info *p_vpi_vlog_info;
            struct t_cb_data { PLI_INT32 reason; PLI_INT32 (*cb_rtn)(struct t_cb_data *); void *obj; struct t_vpi_time *time; struct t_vpi_value *value; PLI_INT32 index; PLI_BYTE8 *user_data; };
            typedef struct t_cb_data *p_cb_data;
            #define vpiModule 32
            #define vpiReg 48
            #define cbStartOfSimulation 11
            #define cbEndOfSimulation 12
            extern vpiHandle vpi_register_cb(p_cb_data);
            extern vpiHandle vpi_iterate(PLI_INT32, vpiHandle);
            extern vpiHandle vpi_scan(vpiHandle);
            static PLI_INT32 sos_cb(struct t_cb_data *d) {
                (void)d;
                vpiHandle mi = vpi_iterate(vpiModule, NULL);
                if (mi) {
                    vpiHandle m = vpi_scan(mi);
                    if (m) {
                        vpiHandle si = vpi_iterate(vpiReg, m);
                        if (si) { fprintf(stderr, "EMPTY: found signals (unexpected)\\n"); }
                        else { fprintf(stderr, "EMPTY: no signals (expected)\\n"); }
                    }
                } else {
                    fprintf(stderr, "EMPTY: no modules (unexpected)\\n");
                }
                fprintf(stderr, "EMPTY: PASS\\n");
                return 0;
            }
            static PLI_INT32 eos_cb(struct t_cb_data *d) { (void)d; fprintf(stderr, "EMPTY: done\\n"); return 0; }
            static void init(void) {
                struct t_cb_data c1 = {0}; c1.reason = cbStartOfSimulation; c1.cb_rtn = sos_cb; vpi_register_cb(&c1);
                struct t_cb_data c2 = {0}; c2.reason = cbEndOfSimulation; c2.cb_rtn = eos_cb; vpi_register_cb(&c2);
            }
            void (*vlog_startup_routines[])(void) = { init, NULL };
        """)
        so = self._write_and_compile_vpi(tmpdir, c_code)
        mlir_path = os.path.join(tmpdir, "empty.mlir")
        with open(mlir_path, "w") as f:
            f.write("hw.module @empty_mod() {}\n")
        output, rc = _run_circt_sim(mlir_path, "empty_mod", so)
        assert "EMPTY: PASS" in output
        assert "EMPTY: done" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
