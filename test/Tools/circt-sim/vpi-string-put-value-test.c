// VPI string put_value test for signal-backed string ports.
// Exercises vpi_put_value(vpiStringVal) -> HDL propagation on !llvm.struct<(ptr,i64)>.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

#define vpiIntVal 6
#define vpiStringVal 8
#define vpiSimTime 2
#define cbAfterDelay 9
#define cbStartOfSimulation 11
#define cbEndOfSimulation 12

extern vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope);
extern void vpi_get_value(vpiHandle expr, p_vpi_value value_p);
extern void vpi_put_value(vpiHandle object, p_vpi_value value_p,
                          p_vpi_time time_p, PLI_INT32 flags);
extern vpiHandle vpi_register_cb(p_cb_data cb_data_p);

static int tests_passed = 0;
static int tests_failed = 0;
static int expected_sum = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      tests_passed++;                                                          \
    } else {                                                                   \
      tests_failed++;                                                          \
      fprintf(stderr, "FAIL: %s\n", msg);                                     \
    }                                                                          \
  } while (0)

static void register_after_delay(PLI_UINT32 delayPs,
                                 PLI_INT32 (*cb)(struct t_cb_data *)) {
  struct t_vpi_time delay;
  memset(&delay, 0, sizeof(delay));
  delay.type = vpiSimTime;
  delay.low = delayPs;

  struct t_cb_data cb_data;
  memset(&cb_data, 0, sizeof(cb_data));
  cb_data.reason = cbAfterDelay;
  cb_data.time = &delay;
  cb_data.cb_rtn = cb;
  vpi_register_cb(&cb_data);
}

static PLI_INT32 check_sum_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  vpiHandle sumSig =
      vpi_handle_by_name("vpi_string_test.stream_in_string_asciival_sum", NULL);
  CHECK(sumSig != NULL, "found stream_in_string_asciival_sum (delayed)");
  if (!sumSig)
    return 0;

  struct t_vpi_value readSum;
  memset(&readSum, 0, sizeof(readSum));
  readSum.format = vpiIntVal;
  vpi_get_value(sumSig, &readSum);
  CHECK(readSum.value.integer == expected_sum,
        "string write propagated to asciival_sum");
  return 0;
}

static PLI_INT32 run_test_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  const char *teststr = "\x1b[33myellow\x1b[49m\x1b[39m";
  expected_sum = 0;
  for (const char *p = teststr; *p; ++p)
    expected_sum += (unsigned char)*p;

  vpiHandle strSig = vpi_handle_by_name("vpi_string_test.stream_in_string", NULL);
  vpiHandle sumSig =
      vpi_handle_by_name("vpi_string_test.stream_in_string_asciival_sum", NULL);
  CHECK(strSig != NULL, "found stream_in_string");
  CHECK(sumSig != NULL, "found stream_in_string_asciival_sum");
  if (!strSig || !sumSig)
    return 0;

  struct t_vpi_value writeVal;
  memset(&writeVal, 0, sizeof(writeVal));
  writeVal.format = vpiStringVal;
  writeVal.value.str = (PLI_BYTE8 *)teststr;
  vpi_put_value(strSig, &writeVal, NULL, 0);

  struct t_vpi_value readStr;
  memset(&readStr, 0, sizeof(readStr));
  readStr.format = vpiStringVal;
  vpi_get_value(strSig, &readStr);
  CHECK(readStr.value.str != NULL, "string readback is non-null");
  if (readStr.value.str)
    CHECK(strcmp((const char *)readStr.value.str, teststr) == 0,
          "string readback matches write");

  // Check HDL-observable effect after the event queue advances.
  register_after_delay(/*delayPs=*/10, check_sum_cb);

  fprintf(stderr, "VPI_STRING: %d passed, %d failed\n", tests_passed,
          tests_failed);
  return 0;
}

static PLI_INT32 start_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  // Run the write+read check after startup so always @(...) waiters are armed.
  register_after_delay(/*delayPs=*/10, run_test_cb);
  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_STRING: FINAL: %d passed, %d failed\n", tests_passed,
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
