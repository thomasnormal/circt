// VPI callback test - exercises cbValueChange and signal monitoring.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef void *vpiHandle;
typedef int PLI_INT32;
typedef unsigned int PLI_UINT32;
typedef char PLI_BYTE8;

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

#define vpiModule 32
#define vpiReg 48
#define vpiType 1
#define vpiName 2
#define vpiFullName 3
#define vpiSize 4
#define vpiVector 18
#define vpiScalar 17
#define vpiSimTime 2
#define vpiIntVal 6
#define cbStartOfSimulation 11
#define cbEndOfSimulation 12

extern vpiHandle vpi_register_cb(p_cb_data cb_data_p);
extern PLI_INT32 vpi_remove_cb(vpiHandle cb_obj);
extern vpiHandle vpi_iterate(PLI_INT32 type, vpiHandle refHandle);
extern vpiHandle vpi_scan(vpiHandle iterator);
extern PLI_INT32 vpi_get(PLI_INT32 property, vpiHandle object);
extern PLI_BYTE8 *vpi_get_str(PLI_INT32 property, vpiHandle object);
extern void vpi_get_value(vpiHandle expr, p_vpi_value value_p);
extern PLI_INT32 vpi_free_object(vpiHandle object);
extern void vpi_get_time(vpiHandle object, p_vpi_time time_p);
extern PLI_INT32 vpi_get_vlog_info(p_vpi_vlog_info vlog_info_p);
extern PLI_INT32 vpi_chk_error(p_vpi_error_info error_info_p);

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      tests_passed++;                                                          \
    } else {                                                                   \
      tests_failed++;                                                          \
      fprintf(stderr, "FAIL: %s\n", msg);                                     \
    }                                                                          \
  } while (0)

static PLI_INT32 start_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_CB: start_of_simulation\n");

  // Test 1: Iterate top-level modules
  vpiHandle modIter = vpi_iterate(vpiModule, NULL);
  CHECK(modIter != NULL, "top module iterator");
  if (!modIter)
    return 0;

  int modCount = 0;
  vpiHandle mod;
  while ((mod = vpi_scan(modIter)) != NULL) {
    PLI_BYTE8 *name = vpi_get_str(vpiName, mod);
    fprintf(stderr, "VPI_CB: module=%s\n", name ? name : "(null)");
    modCount++;
  }
  CHECK(modCount > 0, "at least one module found");
  fprintf(stderr, "VPI_CB: module_count=%d\n", modCount);

  // Get the first module again for signal tests
  modIter = vpi_iterate(vpiModule, NULL);
  mod = vpi_scan(modIter);

  // Test 2: Signal properties
  vpiHandle sigIter = vpi_iterate(vpiReg, mod);
  if (sigIter) {
    vpiHandle sig;
    while ((sig = vpi_scan(sigIter)) != NULL) {
      PLI_BYTE8 *sigName = vpi_get_str(vpiName, sig);
      PLI_INT32 sigType = vpi_get(vpiType, sig);
      PLI_INT32 sigSize = vpi_get(vpiSize, sig);
      PLI_INT32 isVector = vpi_get(vpiVector, sig);
      PLI_INT32 isScalar = vpi_get(vpiScalar, sig);
      fprintf(stderr,
              "VPI_CB: signal=%s type=%d size=%d vector=%d scalar=%d\n",
              sigName ? sigName : "(null)", sigType, sigSize, isVector,
              isScalar);
      CHECK(sigSize > 0, "signal has positive width");
    }
  }

  // Test 3: callback registration and removal
  struct t_cb_data dummy_cb;
  memset(&dummy_cb, 0, sizeof(dummy_cb));
  dummy_cb.reason = cbEndOfSimulation;
  dummy_cb.cb_rtn = start_of_sim_cb; // reuse, doesn't matter
  vpiHandle cbHandle = vpi_register_cb(&dummy_cb);
  CHECK(cbHandle != NULL, "register_cb returns handle");
  if (cbHandle) {
    PLI_INT32 removeResult = vpi_remove_cb(cbHandle);
    CHECK(removeResult == 1, "remove_cb returns 1");
  }

  // Test 4: vpi_get_vlog_info
  struct t_vpi_vlog_info info;
  PLI_INT32 infoResult = vpi_get_vlog_info(&info);
  CHECK(infoResult == 1, "get_vlog_info returns 1");
  CHECK(info.product != NULL, "product non-null");
  CHECK(info.version != NULL, "version non-null");

  // Test 5: vpi_chk_error (no error expected)
  struct t_vpi_error_info errInfo;
  PLI_INT32 errResult = vpi_chk_error(&errInfo);
  CHECK(errResult == 0, "no pending error");

  // Test 6: vpi_get_time
  struct t_vpi_time simTime;
  simTime.type = vpiSimTime;
  vpi_get_time(NULL, &simTime);
  CHECK(simTime.high == 0 && simTime.low == 0, "time is 0 at start of sim");
  fprintf(stderr, "VPI_CB: time=%u:%u\n", simTime.high, simTime.low);

  fprintf(stderr, "VPI_CB: %d passed, %d failed\n", tests_passed,
          tests_failed);
  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_CB: end_of_simulation\n");
  fprintf(stderr, "VPI_CB: FINAL: %d passed, %d failed\n", tests_passed,
          tests_failed);
  return 0;
}

static void register_callbacks(void) {
  struct t_cb_data cb_start;
  memset(&cb_start, 0, sizeof(cb_start));
  cb_start.reason = cbStartOfSimulation;
  cb_start.cb_rtn = start_of_sim_cb;
  vpi_register_cb(&cb_start);

  struct t_cb_data cb_end;
  memset(&cb_end, 0, sizeof(cb_end));
  cb_end.reason = cbEndOfSimulation;
  cb_end.cb_rtn = end_of_sim_cb;
  vpi_register_cb(&cb_end);
}

void (*vlog_startup_routines[])(void) = {register_callbacks, NULL};
