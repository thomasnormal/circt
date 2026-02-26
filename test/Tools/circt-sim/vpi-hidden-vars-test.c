// VPI visibility test for internal/unused variables.
// Ensures circt-verilog + circt-sim expose hidden regs through VPI handles.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void *vpiHandle;
typedef int PLI_INT32;
typedef unsigned int PLI_UINT32;
typedef char PLI_BYTE8;

struct t_cb_data {
  PLI_INT32 reason;
  PLI_INT32 (*cb_rtn)(struct t_cb_data *);
  void *obj;
  void *time;
  void *value;
  PLI_INT32 index;
  PLI_BYTE8 *user_data;
};
typedef struct t_cb_data *p_cb_data;

#define cbStartOfSimulation 11
#define cbEndOfSimulation 12

extern vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope);
extern vpiHandle vpi_register_cb(p_cb_data cb_data_p);

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      tests_passed++;                                                          \
    } else {                                                                   \
      tests_failed++;                                                          \
      fprintf(stderr, "FAIL: %s\\n", msg);                                  \
    }                                                                          \
  } while (0)

static void check_handle(const char *name) {
  vpiHandle h = vpi_handle_by_name(name, NULL);
  CHECK(h != NULL, name);
}

static PLI_INT32 start_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;

  check_handle("vpi_hidden_vars.a");
  check_handle("vpi_hidden_vars.b");
  check_handle("vpi_hidden_vars._underscore_name");

  // Check both escaped and normalized forms for extended identifiers.
  check_handle("vpi_hidden_vars.weird.signal[1]");
  check_handle("vpi_hidden_vars.weird.signal[2]");
  check_handle("vpi_hidden_vars.\\weird.signal[1] ");
  check_handle("vpi_hidden_vars.\\weird.signal[2] ");

  fprintf(stderr, "VPI_HIDDEN: %d passed, %d failed\\n", tests_passed,
          tests_failed);
  return 0;
}

static PLI_INT32 end_of_sim_cb(struct t_cb_data *cb_data) {
  (void)cb_data;
  fprintf(stderr, "VPI_HIDDEN: FINAL: %d passed, %d failed\\n", tests_passed,
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
