// VPI put_value masking test for narrow signals.
// Verifies that writing a value wider than the signal width
// doesn't crash (APInt assertion) and properly masks the value.

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
#define vpiSize 4
#define vpiScalarVal 5
#define vpiIntVal 6
#define vpiVectorVal 9
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

  // Find the narrow_sel signal (2-bit)
  vpiHandle sig = vpi_handle_by_name("narrow_test.narrow_sel", NULL);
  CHECK(sig != NULL, "found narrow_sel signal");
  if (!sig)
    return 0;

  PLI_INT32 width = vpi_get(vpiSize, sig);
  fprintf(stderr, "VPI_NARROW: signal width=%d\n", width);
  CHECK(width == 2, "narrow_sel width is 2");

  // Write 0xFF (255) to a 2-bit signal via vpiIntVal.
  // This should NOT crash — must mask to 2 bits (0xFF & 0x3 = 3).
  struct t_vpi_value writeVal;
  writeVal.format = vpiIntVal;
  writeVal.value.integer = 0xFF;  // 255, wider than 2 bits
  vpi_put_value(sig, &writeVal, NULL, 0);

  struct t_vpi_value readBack;
  readBack.format = vpiIntVal;
  vpi_get_value(sig, &readBack);
  fprintf(stderr, "VPI_NARROW: wrote=0xFF read=%d\n", readBack.value.integer);
  CHECK(readBack.value.integer == 3, "masked to 2 bits: 0xFF -> 3");

  // Write -1 (all bits set) — should also mask to 3.
  writeVal.value.integer = -1;
  vpi_put_value(sig, &writeVal, NULL, 0);
  vpi_get_value(sig, &readBack);
  fprintf(stderr, "VPI_NARROW: wrote=-1 read=%d\n", readBack.value.integer);
  CHECK(readBack.value.integer == 3, "masked to 2 bits: -1 -> 3");

  // Write 2 — should stay as 2.
  writeVal.value.integer = 2;
  vpi_put_value(sig, &writeVal, NULL, 0);
  vpi_get_value(sig, &readBack);
  fprintf(stderr, "VPI_NARROW: wrote=2 read=%d\n", readBack.value.integer);
  CHECK(readBack.value.integer == 2, "value 2 fits in 2 bits");

  // Test vpiVectorVal with oversized aval
  struct t_vpi_vecval vec;
  vec.aval = 0xDEADBEEF;  // Way wider than 2 bits
  vec.bval = 0;
  struct t_vpi_value vecVal;
  vecVal.format = vpiVectorVal;
  vecVal.value.vector = &vec;
  vpi_put_value(sig, &vecVal, NULL, 0);
  vpi_get_value(sig, &readBack);
  fprintf(stderr, "VPI_NARROW: vecval=0xDEADBEEF read=%d\n",
          readBack.value.integer);
  CHECK(readBack.value.integer == 3, "vecval masked: 0xDEADBEEF & 3 = 3");

  fprintf(stderr, "VPI_NARROW: %d passed, %d failed\n", tests_passed,
          tests_failed);
  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_NARROW: FINAL: %d passed, %d failed\n", tests_passed,
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
