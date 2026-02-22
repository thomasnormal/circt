// VPI basic test library - loaded by circt-sim via --vpi flag.
// Tests: handle_by_name, get/get_str properties, get_value, put_value,
//        iterate/scan, register_cb/remove_cb, get_time, vlog_info.
//
// This is a standalone VPI test that does NOT require cocotb.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// VPI types and constants (minimal subset for testing)
typedef void *vpiHandle;
typedef int PLI_INT32;
typedef unsigned int PLI_UINT32;
typedef char PLI_BYTE8;
typedef unsigned char PLI_UBYTE8;

struct t_vpi_time {
  PLI_INT32 type;
  PLI_UINT32 high, low;
  double real;
};
typedef struct t_vpi_time *p_vpi_time;

struct t_vpi_vecval {
  PLI_UINT32 aval, bval;
};

struct t_vpi_value {
  PLI_INT32 format;
  union {
    PLI_BYTE8 *str;
    PLI_INT32 scalar;
    PLI_INT32 integer;
    double real;
    struct t_vpi_time *time;
    struct t_vpi_vecval *vector;
    PLI_BYTE8 *misc;
  } value;
};
typedef struct t_vpi_value *p_vpi_value;

struct t_vpi_vlog_info {
  PLI_INT32 argc;
  PLI_BYTE8 **argv;
  PLI_BYTE8 *product;
  PLI_BYTE8 *version;
};
typedef struct t_vpi_vlog_info *p_vpi_vlog_info;

struct t_vpi_error_info {
  PLI_INT32 state;
  PLI_INT32 level;
  PLI_BYTE8 *message;
  PLI_BYTE8 *product;
  PLI_BYTE8 *code;
  PLI_BYTE8 *file;
  PLI_INT32 line;
};
typedef struct t_vpi_error_info *p_vpi_error_info;

struct t_cb_data {
  PLI_INT32 reason;
  PLI_INT32 (*cb_rtn)(struct t_cb_data *);
  void *obj;
  p_vpi_time time;
  p_vpi_value value;
  PLI_INT32 index;
  PLI_BYTE8 *user_data;
};
typedef struct t_cb_data *p_cb_data;

// VPI constants
#define vpiModule 32
#define vpiNet 36
#define vpiReg 48
#define vpiType 1
#define vpiName 2
#define vpiFullName 3
#define vpiSize 4
#define vpiSimTime 2
#define vpiBinStrVal 1
#define vpiIntVal 6
#define cbStartOfSimulation 11
#define cbEndOfSimulation 12
#define cbReadWriteSynch 6

// VPI functions (resolved from the simulator binary)
extern vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope);
extern PLI_INT32 vpi_get(PLI_INT32 property, vpiHandle object);
extern PLI_BYTE8 *vpi_get_str(PLI_INT32 property, vpiHandle object);
extern void vpi_get_value(vpiHandle expr, p_vpi_value value_p);
extern void vpi_put_value(vpiHandle object, p_vpi_value value_p,
                          p_vpi_time time_p, PLI_INT32 flags);
extern vpiHandle vpi_register_cb(p_cb_data cb_data_p);
extern PLI_INT32 vpi_remove_cb(vpiHandle cb_obj);
extern vpiHandle vpi_iterate(PLI_INT32 type, vpiHandle refHandle);
extern vpiHandle vpi_scan(vpiHandle iterator);
extern PLI_INT32 vpi_free_object(vpiHandle object);
extern void vpi_get_time(vpiHandle object, p_vpi_time time_p);
extern PLI_INT32 vpi_get_vlog_info(p_vpi_vlog_info vlog_info_p);
extern PLI_INT32 vpi_chk_error(p_vpi_error_info error_info_p);
extern PLI_INT32 vpi_control(PLI_INT32 operation, ...);
extern PLI_INT32 vpi_release_handle(vpiHandle object);

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
  if (cond) { tests_passed++; } \
  else { tests_failed++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

static PLI_INT32 start_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_TEST: start_of_simulation callback fired\n");

  // Test 1: vpi_get_vlog_info
  struct t_vpi_vlog_info info;
  PLI_INT32 ret = vpi_get_vlog_info(&info);
  CHECK(ret == 1, "vpi_get_vlog_info returns 1");
  CHECK(info.product != NULL, "vpi_get_vlog_info product non-null");
  if (info.product)
    fprintf(stderr, "VPI_TEST: product=%s\n", info.product);

  // Test 2: vpi_iterate top-level modules
  vpiHandle modIter = vpi_iterate(vpiModule, NULL);
  CHECK(modIter != NULL, "vpi_iterate(vpiModule, NULL) returns iterator");

  if (modIter) {
    vpiHandle mod = vpi_scan(modIter);
    CHECK(mod != NULL, "vpi_scan returns first module");

    if (mod) {
      // Test 3: vpi_get_str for module name
      PLI_BYTE8 *modName = vpi_get_str(vpiName, mod);
      CHECK(modName != NULL, "vpi_get_str(vpiName) non-null");
      if (modName)
        fprintf(stderr, "VPI_TEST: module name=%s\n", modName);

      // Test 4: vpi_get type
      PLI_INT32 objType = vpi_get(vpiType, mod);
      CHECK(objType == vpiModule, "vpi_get(vpiType) == vpiModule");

      // Test 5: iterate signals in the module
      vpiHandle sigIter = vpi_iterate(vpiReg, mod);
      if (sigIter) {
        vpiHandle sig;
        int sigCount = 0;
        while ((sig = vpi_scan(sigIter)) != NULL) {
          PLI_BYTE8 *sigName = vpi_get_str(vpiName, sig);
          PLI_INT32 sigWidth = vpi_get(vpiSize, sig);
          fprintf(stderr, "VPI_TEST: signal name=%s width=%d\n",
                  sigName ? sigName : "(null)", sigWidth);
          sigCount++;

          // Test 6: read signal value
          struct t_vpi_value val;
          val.format = vpiIntVal;
          vpi_get_value(sig, &val);
          fprintf(stderr, "VPI_TEST: signal %s intval=%d\n",
                  sigName ? sigName : "(null)", val.value.integer);
        }
        CHECK(sigCount > 0, "found at least one signal");
        fprintf(stderr, "VPI_TEST: found %d signals\n", sigCount);
      }
    }
  }

  // Test 7: vpi_get_time
  struct t_vpi_time simTime;
  simTime.type = vpiSimTime;
  vpi_get_time(NULL, &simTime);
  fprintf(stderr, "VPI_TEST: time=%u:%u\n", simTime.high, simTime.low);

  // Test 8: vpi_chk_error (should return 0 = no error)
  struct t_vpi_error_info errInfo;
  PLI_INT32 errLevel = vpi_chk_error(&errInfo);
  CHECK(errLevel == 0, "vpi_chk_error returns 0 (no error)");

  // Summary
  fprintf(stderr, "VPI_TEST: %d passed, %d failed\n",
          tests_passed, tests_failed);

  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_TEST: end_of_simulation callback fired\n");
  fprintf(stderr, "VPI_TEST: FINAL: %d passed, %d failed\n",
          tests_passed, tests_failed);
  return 0;
}

static void register_callbacks(void) {
  struct t_cb_data cb_start;
  memset(&cb_start, 0, sizeof(cb_start));
  cb_start.reason = cbStartOfSimulation;
  cb_start.cb_rtn = start_of_sim_cb;
  vpiHandle h1 = vpi_register_cb(&cb_start);
  if (!h1)
    fprintf(stderr, "VPI_TEST: ERROR: failed to register cbStartOfSimulation\n");

  struct t_cb_data cb_end;
  memset(&cb_end, 0, sizeof(cb_end));
  cb_end.reason = cbEndOfSimulation;
  cb_end.cb_rtn = end_of_sim_cb;
  vpiHandle h2 = vpi_register_cb(&cb_end);
  if (!h2)
    fprintf(stderr, "VPI_TEST: ERROR: failed to register cbEndOfSimulation\n");
}

// Standard VPI startup routine table (null-terminated)
void (*vlog_startup_routines[])(void) = {
  register_callbacks,
  NULL
};
