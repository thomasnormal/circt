#!/usr/bin/env python3
"""Compare two VCD waveforms by normalized per-signal event streams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TIMESCALE_UNITS_TO_FS = {
    "fs": 1,
    "ps": 1_000,
    "ns": 1_000_000,
    "us": 1_000_000_000,
    "ms": 1_000_000_000_000,
    "s": 1_000_000_000_000_000,
}


@dataclass
class SignalDigest:
    digest_hex: str
    changes: int
    first_time_fs: int | None
    last_time_fs: int | None
    last_value: str | None


@dataclass
class ParseResult:
    path: Path
    timescale_fs: int
    signals: dict[str, SignalDigest]
    ambiguous_names: set[str]


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("lhs_vcd", help="First VCD file path.")
    parser.add_argument("rhs_vcd", help="Second VCD file path.")
    parser.add_argument(
        "--label-lhs",
        default="lhs",
        help="Display label for first waveform (default: lhs).",
    )
    parser.add_argument(
        "--label-rhs",
        default="rhs",
        help="Display label for second waveform (default: rhs).",
    )
    parser.add_argument(
        "--signal",
        action="append",
        default=[],
        help="Exact normalized signal name to compare (repeatable).",
    )
    parser.add_argument(
        "--signal-regex",
        action="append",
        default=[],
        help="Regex over normalized signal names to compare (repeatable).",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help=(
            "Strip this prefix from signal names before matching "
            "(repeatable; useful for root-scope differences)."
        ),
    )
    parser.add_argument(
        "--require-same-signal-set",
        action="store_true",
        help="Fail if normalized signal sets differ.",
    )
    parser.add_argument(
        "--max-report-signals",
        type=int,
        default=20,
        help="Maximum mismatching/missing signals reported (default: 20).",
    )
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Optional TSV output path for per-signal comparison rows.",
    )
    parser.add_argument(
        "--out-summary-tsv",
        default="",
        help="Optional TSV output path for one-line summary.",
    )
    parser.add_argument(
        "--ignore-timescale",
        action="store_true",
        help="Do not fail on timescale mismatch (timestamps are still normalized to fs).",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when mismatches are present.",
    )
    return parser.parse_args()


def parse_timescale_blob(blob: str) -> int | None:
    # Accept forms like "$timescale 1ns $end" and "$timescale 10 ps $end".
    match = re.search(r"([0-9]+)\s*([fpnum]?s)\b", blob)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    unit_fs = TIMESCALE_UNITS_TO_FS.get(unit)
    if unit_fs is None:
        return None
    return value * unit_fs


def normalize_name(name: str, strip_prefixes: list[str]) -> str:
    out = name
    for prefix in strip_prefixes:
        p = prefix.strip()
        if not p:
            continue
        if out == p:
            out = ""
            continue
        dotted = f"{p}."
        if out.startswith(dotted):
            out = out[len(dotted) :]
    return out


def should_track_signal(
    name: str, exact: set[str], regex_rules: list[re.Pattern[str]]
) -> bool:
    if not exact and not regex_rules:
        return True
    if name in exact:
        return True
    return any(rule.search(name) for rule in regex_rules)


def parse_vcd(
    path: Path,
    strip_prefixes: list[str],
    exact_filters: set[str],
    regex_filters: list[re.Pattern[str]],
) -> ParseResult:
    if not path.is_file():
        fail(f"VCD file not found: {path}")

    id_to_name: dict[str, str] = {}
    used_names: dict[str, str] = {}
    ambiguous_names: set[str] = set()
    scope_stack: list[str] = []
    timescale_fs = 1
    header_done = False
    current_time_fs = 0

    # Mutable state for each tracked signal.
    signal_hash: dict[str, hashlib._Hash] = {}
    signal_changes: dict[str, int] = {}
    signal_first_time: dict[str, int | None] = {}
    signal_last_time: dict[str, int | None] = {}
    signal_last_value: dict[str, str | None] = {}

    with path.open(encoding="utf-8", errors="replace") as handle:
        it = iter(handle)
        for raw in it:
            line = raw.strip()
            if not line:
                continue

            if not header_done:
                if line.startswith("$scope"):
                    parts = line.split()
                    if len(parts) >= 3:
                        scope_stack.append(parts[2])
                    continue
                if line.startswith("$upscope"):
                    if scope_stack:
                        scope_stack.pop()
                    continue
                if line.startswith("$timescale"):
                    blob = line
                    while "$end" not in blob:
                        try:
                            blob += " " + next(it).strip()
                        except StopIteration:
                            break
                    parsed = parse_timescale_blob(blob)
                    if parsed is not None:
                        timescale_fs = parsed
                    continue
                if line.startswith("$var"):
                    parts = line.split()
                    # $var <type> <size> <id> <ref> [index] $end
                    if len(parts) < 5:
                        continue
                    vcd_id = parts[3]
                    ref = parts[4]
                    if ref.startswith("\\"):
                        ref = ref[1:]
                    full_name = ".".join([*scope_stack, ref]) if scope_stack else ref
                    norm_name = normalize_name(full_name, strip_prefixes)
                    if not norm_name:
                        continue
                    if not should_track_signal(norm_name, exact_filters, regex_filters):
                        continue

                    # VCD identifiers should be unique, but normalized names can
                    # collide after prefix stripping. Mark as ambiguous and skip.
                    if norm_name in used_names and used_names[norm_name] != vcd_id:
                        ambiguous_names.add(norm_name)
                    used_names[norm_name] = vcd_id
                    id_to_name[vcd_id] = norm_name
                    continue
                if line.startswith("$enddefinitions"):
                    header_done = True
                    # Remove ambiguous names and their IDs from tracking.
                    if ambiguous_names:
                        keep = {}
                        for vcd_id, name in id_to_name.items():
                            if name not in ambiguous_names:
                                keep[vcd_id] = name
                        id_to_name = keep
                    for name in set(id_to_name.values()):
                        signal_hash[name] = hashlib.sha256()
                        signal_changes[name] = 0
                        signal_first_time[name] = None
                        signal_last_time[name] = None
                        signal_last_value[name] = None
                    continue
                continue

            # Runtime section.
            if line.startswith("#"):
                try:
                    current_time_fs = int(line[1:].strip()) * timescale_fs
                except ValueError:
                    continue
                continue
            if line.startswith("$"):
                # Ignore $dumpvars/$end/etc.
                continue

            value: str | None = None
            vcd_id: str | None = None
            lead = line[0]
            if lead in "01xXzZ":
                value = lead.lower()
                vcd_id = line[1:].strip()
            elif lead in "bBrRsS":
                payload = line[1:].strip()
                parts = payload.split(None, 1)
                if len(parts) == 2:
                    value = parts[0].lower()
                    vcd_id = parts[1].strip()

            if not value or not vcd_id:
                continue
            name = id_to_name.get(vcd_id)
            if name is None:
                continue

            signal_hash[name].update(f"{current_time_fs}\t{value}\n".encode("utf-8"))
            signal_changes[name] += 1
            if signal_first_time[name] is None:
                signal_first_time[name] = current_time_fs
            signal_last_time[name] = current_time_fs
            signal_last_value[name] = value

    signals: dict[str, SignalDigest] = {}
    for name, hasher in signal_hash.items():
        signals[name] = SignalDigest(
            digest_hex=hasher.hexdigest(),
            changes=signal_changes[name],
            first_time_fs=signal_first_time[name],
            last_time_fs=signal_last_time[name],
            last_value=signal_last_value[name],
        )

    return ParseResult(
        path=path,
        timescale_fs=timescale_fs,
        signals=signals,
        ambiguous_names=ambiguous_names,
    )


def emit_signal_tsv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["signal", "status", "lhs_digest", "rhs_digest", "note"])
        writer.writerows(rows)


def emit_summary_tsv(
    path: Path,
    label_lhs: str,
    label_rhs: str,
    lhs_path: Path,
    rhs_path: Path,
    summary: dict[str, int | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "lhs_label",
                "rhs_label",
                "lhs_vcd",
                "rhs_vcd",
                "lhs_signals",
                "rhs_signals",
                "signals_compared",
                "signals_only_lhs",
                "signals_only_rhs",
                "signals_mismatched",
                "ambiguous_names_lhs",
                "ambiguous_names_rhs",
                "timescale_fs_lhs",
                "timescale_fs_rhs",
                "timescale_match",
            ]
        )
        writer.writerow(
            [
                label_lhs,
                label_rhs,
                str(lhs_path),
                str(rhs_path),
                summary["lhs_signals"],
                summary["rhs_signals"],
                summary["signals_compared"],
                summary["signals_only_lhs"],
                summary["signals_only_rhs"],
                summary["signals_mismatched"],
                summary["ambiguous_names_lhs"],
                summary["ambiguous_names_rhs"],
                summary["timescale_fs_lhs"],
                summary["timescale_fs_rhs"],
                summary["timescale_match"],
            ]
        )


def main() -> None:
    args = parse_args()
    lhs_path = Path(args.lhs_vcd).resolve()
    rhs_path = Path(args.rhs_vcd).resolve()

    regex_filters: list[re.Pattern[str]] = []
    for expr in args.signal_regex:
        try:
            regex_filters.append(re.compile(expr))
        except re.error as exc:
            fail(f"invalid --signal-regex '{expr}': {exc}")

    exact_filters = {s for s in args.signal if s}
    lhs = parse_vcd(lhs_path, args.strip_prefix, exact_filters, regex_filters)
    rhs = parse_vcd(rhs_path, args.strip_prefix, exact_filters, regex_filters)

    lhs_names = set(lhs.signals.keys())
    rhs_names = set(rhs.signals.keys())
    only_lhs = sorted(lhs_names - rhs_names)
    only_rhs = sorted(rhs_names - lhs_names)
    compared = sorted(lhs_names & rhs_names)

    timescale_match = lhs.timescale_fs == rhs.timescale_fs
    if not args.ignore_timescale and not timescale_match:
        print(
            f"timescale mismatch: {args.label_lhs}={lhs.timescale_fs}fs "
            f"{args.label_rhs}={rhs.timescale_fs}fs",
            file=sys.stderr,
        )

    signal_rows: list[tuple[str, str, str, str, str]] = []
    mismatched: list[str] = []

    for name in only_lhs:
        signal_rows.append((name, "only_lhs", lhs.signals[name].digest_hex, "-", ""))
    for name in only_rhs:
        signal_rows.append((name, "only_rhs", "-", rhs.signals[name].digest_hex, ""))
    for name in compared:
        l = lhs.signals[name]
        r = rhs.signals[name]
        status = "match"
        note = ""
        if l.digest_hex != r.digest_hex:
            status = "mismatch"
            note = (
                f"lhs_changes={l.changes},rhs_changes={r.changes},"
                f"lhs_last={l.last_time_fs}:{l.last_value},"
                f"rhs_last={r.last_time_fs}:{r.last_value}"
            )
            mismatched.append(name)
        signal_rows.append((name, status, l.digest_hex, r.digest_hex, note))

    summary = {
        "lhs_signals": len(lhs_names),
        "rhs_signals": len(rhs_names),
        "signals_compared": len(compared),
        "signals_only_lhs": len(only_lhs),
        "signals_only_rhs": len(only_rhs),
        "signals_mismatched": len(mismatched),
        "ambiguous_names_lhs": len(lhs.ambiguous_names),
        "ambiguous_names_rhs": len(rhs.ambiguous_names),
        "timescale_fs_lhs": lhs.timescale_fs,
        "timescale_fs_rhs": rhs.timescale_fs,
        "timescale_match": "1" if timescale_match else "0",
    }

    if args.out_tsv:
        emit_signal_tsv(Path(args.out_tsv).resolve(), signal_rows)
    if args.out_summary_tsv:
        emit_summary_tsv(
            Path(args.out_summary_tsv).resolve(),
            args.label_lhs,
            args.label_rhs,
            lhs_path,
            rhs_path,
            summary,
        )

    print(
        f"{args.label_lhs}_signals={summary['lhs_signals']} "
        f"{args.label_rhs}_signals={summary['rhs_signals']} "
        f"signals_compared={summary['signals_compared']} "
        f"signals_only_{args.label_lhs}={summary['signals_only_lhs']} "
        f"signals_only_{args.label_rhs}={summary['signals_only_rhs']} "
        f"signals_mismatched={summary['signals_mismatched']} "
        f"ambiguous_{args.label_lhs}={summary['ambiguous_names_lhs']} "
        f"ambiguous_{args.label_rhs}={summary['ambiguous_names_rhs']} "
        f"timescale_match={summary['timescale_match']}"
    )

    if only_lhs:
        listed = ", ".join(only_lhs[: args.max_report_signals])
        suffix = "" if len(only_lhs) <= args.max_report_signals else ", ..."
        print(f"signals only in {args.label_lhs}: {listed}{suffix}", file=sys.stderr)
    if only_rhs:
        listed = ", ".join(only_rhs[: args.max_report_signals])
        suffix = "" if len(only_rhs) <= args.max_report_signals else ", ..."
        print(f"signals only in {args.label_rhs}: {listed}{suffix}", file=sys.stderr)
    if mismatched:
        listed = ", ".join(mismatched[: args.max_report_signals])
        suffix = "" if len(mismatched) <= args.max_report_signals else ", ..."
        print(f"mismatching signals: {listed}{suffix}", file=sys.stderr)

    mismatches_present = (
        (not timescale_match and not args.ignore_timescale)
        or (args.require_same_signal_set and (only_lhs or only_rhs))
        or bool(mismatched)
    )
    if args.fail_on_mismatch and mismatches_present:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
