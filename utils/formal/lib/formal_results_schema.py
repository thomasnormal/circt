#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared formal results schema constants."""

REQUIRED_FIELDS = (
    "schema_version",
    "suite",
    "mode",
    "case_id",
    "case_path",
    "status",
    "reason_code",
    "stage",
    "solver",
    "solver_time_ms",
    "frontend_time_ms",
    "log_path",
    "artifact_dir",
)

ALLOWED_MODES = {"BMC", "LEC", "CONNECTIVITY_LEC"}

ALLOWED_STATUS = {
    "PASS",
    "FAIL",
    "ERROR",
    "TIMEOUT",
    "UNKNOWN",
    "SKIP",
    "XFAIL",
    "XPASS",
}

ALLOWED_STAGES = {"frontend", "lowering", "solver", "result", "postprocess"}
