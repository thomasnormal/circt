#!/usr/bin/env bash
set -euo pipefail

subcmd="${1:-}"
if [[ "$subcmd" != "cover" ]]; then
  echo "expected cover subcommand" >&2
  exit 9
fi
shift

work_dir=""
args=("$@")

for ((i=0; i<${#args[@]}; ++i)); do
  arg="${args[$i]}"
  case "$arg" in
    --work-dir)
      work_dir="${args[$((i+1))]:-}"
      ;;
    --work-dir=*)
      work_dir="${arg#*=}"
      ;;
    --mutations-yosys|--mutations-yosys=*)
      if [[ "${MUT_FAKE_FAIL_IF_YOSYS:-0}" == "1" ]]; then
        echo "unexpected --mutations-yosys" >&2
        exit 7
      fi
      ;;
  esac
done

if [[ -z "$work_dir" ]]; then
  echo "missing --work-dir" >&2
  exit 8
fi

required_csv="${MUT_FAKE_REQUIRE_FLAGS_CSV:-}"
if [[ -n "$required_csv" ]]; then
  IFS=',' read -r -a required_flags <<< "$required_csv"
  for flag in "${required_flags[@]}"; do
    flag="${flag## }"
    flag="${flag%% }"
    [[ -z "$flag" ]] && continue
    seen=0
    for arg in "${args[@]}"; do
      if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
        seen=1
        break
      fi
    done
    if [[ "$seen" -ne 1 ]]; then
      echo "missing required flag: $flag" >&2
      exit 6
    fi
  done
fi

mkdir -p "$work_dir"
printf 'detected_mutants\t%s\nrelevant_mutants\t%s\nerrors\t%s\n' \
  "${MUT_FAKE_DETECTED:-2}" "${MUT_FAKE_RELEVANT:-2}" "${MUT_FAKE_ERRORS:-0}" > "$work_dir/metrics.tsv"
