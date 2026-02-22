// VPI put_value and handle_by_name test.
// Exercises: vpi_handle_by_name, vpi_put_value, vpi_get_value (integer format),
//            vpi_iterate for signal discovery.

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
#define vpiName 2
#define vpiFullName 3
#define vpiSize 4
#define vpiSimTime 2
#define vpiIntVal 6
#define vpiBinStrVal 1
#define cbStartOfSimulation 11
#define cbEndOfSimulation 12

extern vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope);
extern PLI_INT32 vpi_get(PLI_INT32 property, vpiHandle object);
extern PLI_BYTE8 *vpi_get_str(PLI_INT32 property, vpiHandle object);
extern void vpi_get_value(vpiHandle expr, p_vpi_value value_p);
extern void vpi_put_value(vpiHandle object, p_vpi_value value_p,
                          p_vpi_time time_p, PLI_INT32 flags);
extern vpiHandle vpi_register_cb(p_cb_data cb_data_p);
extern vpiHandle vpi_iterate(PLI_INT32 type, vpiHandle refHandle);
extern vpiHandle vpi_scan(vpiHandle iterator);
extern void vpi_get_time(vpiHandle object, p_vpi_time time_p);

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
  fprintf(stderr, "VPI_PUT: start_of_simulation\n");

  // Find signals by iterating
  vpiHandle modIter = vpi_iterate(vpiModule, NULL);
  CHECK(modIter != NULL, "module iterator non-null");
  if (!modIter)
    return 0;

  vpiHandle mod = vpi_scan(modIter);
  CHECK(mod != NULL, "first module non-null");
  if (!mod)
    return 0;

  // Find a signal wider than 1 bit to test put_value on
  vpiHandle sigIter = vpi_iterate(vpiReg, mod);
  CHECK(sigIter != NULL, "signal iterator non-null");
  if (!sigIter)
    return 0;

  vpiHandle sig = NULL;
  PLI_INT32 sigWidth = 0;
  PLI_BYTE8 *sigName = NULL;
  {
    vpiHandle candidate;
    while ((candidate = vpi_scan(sigIter)) != NULL) {
      PLI_INT32 w = vpi_get(vpiSize, candidate);
      if (w >= 8) {
        sig = candidate;
        sigWidth = w;
        sigName = vpi_get_str(vpiName, candidate);
        break;
      }
    }
  }
  CHECK(sig != NULL, "found signal with width >= 8");
  if (!sig)
    return 0;

  fprintf(stderr, "VPI_PUT: testing signal=%s width=%d\n",
          sigName ? sigName : "(null)", sigWidth);

  // Read initial value
  struct t_vpi_value val;
  val.format = vpiIntVal;
  vpi_get_value(sig, &val);
  fprintf(stderr, "VPI_PUT: initial_value=%d\n", val.value.integer);

  // Write a value
  struct t_vpi_value writeVal;
  writeVal.format = vpiIntVal;
  writeVal.value.integer = 42;
  vpi_put_value(sig, &writeVal, NULL, 0);

  // Read back
  struct t_vpi_value readBack;
  readBack.format = vpiIntVal;
  vpi_get_value(sig, &readBack);
  fprintf(stderr, "VPI_PUT: after_write=%d\n", readBack.value.integer);
  CHECK(readBack.value.integer == 42, "put_value/get_value roundtrip == 42");

  // Test binary string format
  struct t_vpi_value binVal;
  binVal.format = vpiBinStrVal;
  vpi_get_value(sig, &binVal);
  CHECK(binVal.value.str != NULL, "binary string non-null");
  if (binVal.value.str)
    fprintf(stderr, "VPI_PUT: binary=%s\n", binVal.value.str);

  // Test handle_by_name
  PLI_BYTE8 *fullName = vpi_get_str(vpiFullName, sig);
  if (fullName) {
    fprintf(stderr, "VPI_PUT: full_name=%s\n", fullName);
    vpiHandle found = vpi_handle_by_name(fullName, NULL);
    CHECK(found != NULL, "handle_by_name found signal");
    if (found) {
      struct t_vpi_value foundVal;
      foundVal.format = vpiIntVal;
      vpi_get_value(found, &foundVal);
      fprintf(stderr, "VPI_PUT: found_value=%d\n", foundVal.value.integer);
      CHECK(foundVal.value.integer == 42,
            "handle_by_name read matches put_value");
    }
  }

  fprintf(stderr, "VPI_PUT: %d passed, %d failed\n", tests_passed,
          tests_failed);
  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_PUT: FINAL: %d passed, %d failed\n", tests_passed,
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
