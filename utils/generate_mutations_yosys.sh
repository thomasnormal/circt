#!/usr/bin/env bash
# Generate mutation list using yosys mutate -list.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: generate_mutations_yosys.sh [options]

Required:
  --design FILE             Input design (.il/.v/.sv)
  --out FILE                Output mutation list file

Optional:
  --top NAME                Top module name (recommended for .v/.sv)
  --count N                 Number of mutations to generate (default: 1000)
  --seed N                  Random seed for mutate (default: 1)
  --yosys PATH              Yosys executable (default: yosys)
  --cache-dir DIR           Optional cache directory for generated mutation
                            lists (content-addressed by design+options)
  --mode NAME               Mutate mode (repeatable)
                            Concrete: inv,const0,const1,cnot0,cnot1
                            Families:
                              arith,control,balanced,all
                              stuck,invert,connect
  --modes CSV               Comma-separated mutate modes (alternative to repeated --mode)
  --mode-count NAME=COUNT   Explicit mutation count for a mode (repeatable)
  --mode-counts CSV         Comma-separated NAME=COUNT mode allocations
  --mode-weight NAME=WEIGHT Relative weight for a mode (repeatable)
  --mode-weights CSV        Comma-separated NAME=WEIGHT mode weights
  --profile NAME            Named mutation profile (repeatable)
  --profiles CSV            Comma-separated named mutation profiles
  --cfg KEY=VALUE           Mutate config entry (repeatable, becomes -cfg KEY VALUE)
  --cfgs CSV                Comma-separated KEY=VALUE mutate config entries
  --select EXPR             Additional mutate select expression (repeatable)
  --selects CSV             Comma-separated mutate select expressions
  -h, --help                Show help

Output format:
  Each line in --out is "<id> <mutation-spec>" (MCY-compatible list form).
USAGE
}

DESIGN=""
OUT_FILE=""
TOP=""
COUNT=1000
SEED=1
YOSYS_BIN="yosys"
CACHE_DIR=""
MODES_CSV=""
MODE_COUNTS_CSV=""
MODE_WEIGHTS_CSV=""
PROFILES_CSV=""
CFGS_CSV=""
SELECTS_CSV=""
declare -a MODE_LIST=()
declare -a MODE_COUNT_LIST=()
declare -a MODE_WEIGHT_LIST=()
declare -a PROFILE_LIST=()
declare -a CFG_LIST=()
declare -a SELECT_LIST=()
CACHE_LOCK_DIR=""
CACHE_LOCK_STALE_SECONDS=3600
CACHE_LOCK_POLL_SECONDS=0.1
CACHE_LOCK_WAIT_NS=0
CACHE_LOCK_CONTENDED=0

cleanup() {
  if [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR:-}" ]]; then
    rm -rf "$WORK_DIR"
  fi
  if [[ -n "${CACHE_LOCK_DIR:-}" ]]; then
    rm -f "${CACHE_LOCK_DIR}/pid" 2>/dev/null || true
    rmdir "$CACHE_LOCK_DIR" 2>/dev/null || true
    CACHE_LOCK_DIR=""
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --design) DESIGN="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    --top) TOP="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --yosys) YOSYS_BIN="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --mode) MODE_LIST+=("$2"); shift 2 ;;
    --modes) MODES_CSV="$2"; shift 2 ;;
    --mode-count) MODE_COUNT_LIST+=("$2"); shift 2 ;;
    --mode-counts) MODE_COUNTS_CSV="$2"; shift 2 ;;
    --mode-weight) MODE_WEIGHT_LIST+=("$2"); shift 2 ;;
    --mode-weights) MODE_WEIGHTS_CSV="$2"; shift 2 ;;
    --profile) PROFILE_LIST+=("$2"); shift 2 ;;
    --profiles) PROFILES_CSV="$2"; shift 2 ;;
    --cfg) CFG_LIST+=("$2"); shift 2 ;;
    --cfgs) CFGS_CSV="$2"; shift 2 ;;
    --select) SELECT_LIST+=("$2"); shift 2 ;;
    --selects) SELECTS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DESIGN" || -z "$OUT_FILE" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$DESIGN" ]]; then
  echo "Design file not found: $DESIGN" >&2
  exit 1
fi
if [[ ! "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --count value: $COUNT" >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --seed value: $SEED" >&2
  exit 1
fi
if [[ -n "$CACHE_DIR" ]]; then
  mkdir -p "$CACHE_DIR"
fi

hash_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return
  fi
  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 | awk '{print $NF}'
    return
  fi
  python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
}

hash_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    printf "missing\n"
    return
  fi
  hash_stdin < "$file"
}

hash_string() {
  local s="$1"
  printf "%s" "$s" | hash_stdin
}

normalize_epoch_ns() {
  local raw="${1:-0}"
  if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
    raw=0
  fi
  if [[ "$raw" -lt 1000000000000 ]]; then
    raw=$((raw * 1000000000))
  fi
  printf "%s\n" "$raw"
}

current_epoch_ns() {
  local now_raw=""
  now_raw="$(date +%s%N 2>/dev/null || true)"
  if [[ ! "$now_raw" =~ ^[0-9]+$ ]]; then
    now_raw="$(python3 -c 'import time; print(time.time_ns())' 2>/dev/null || true)"
  fi
  if [[ ! "$now_raw" =~ ^[0-9]+$ ]]; then
    now_raw="$(date +%s 2>/dev/null || printf "0")"
  fi
  normalize_epoch_ns "$now_raw"
}

elapsed_ns_since() {
  local start_ns="${1:-0}"
  local end_ns=0
  local delta=0
  end_ns="$(current_epoch_ns)"
  if [[ "$start_ns" =~ ^[0-9]+$ ]] && [[ "$end_ns" =~ ^[0-9]+$ ]] && [[ "$end_ns" -ge "$start_ns" ]]; then
    delta=$((end_ns - start_ns))
  fi
  printf "%s\n" "$delta"
}

SCRIPT_START_NS="$(current_epoch_ns)"
mkdir -p "$(dirname "$OUT_FILE")"

file_mtime_epoch() {
  local path="$1"
  local ts=0
  ts="$(stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || printf "0")"
  if [[ ! "$ts" =~ ^[0-9]+$ ]]; then
    ts=0
  fi
  printf "%s\n" "$ts"
}

release_cache_lock() {
  if [[ -n "${CACHE_LOCK_DIR:-}" ]]; then
    rm -f "${CACHE_LOCK_DIR}/pid" 2>/dev/null || true
    rmdir "$CACHE_LOCK_DIR" 2>/dev/null || true
    CACHE_LOCK_DIR=""
  fi
}

acquire_cache_lock() {
  local cache_path="$1"
  local lock_dir="${cache_path}.lock"
  local now_sec=0
  local lock_mtime=0
  local lock_age=0
  local lock_start_ns=0
  local lock_end_ns=0
  local local_contended=0

  lock_start_ns="$(current_epoch_ns)"
  while true; do
    if mkdir "$lock_dir" 2>/dev/null; then
      CACHE_LOCK_DIR="$lock_dir"
      printf "%s\n" "$$" > "${lock_dir}/pid" 2>/dev/null || true
      lock_end_ns="$(current_epoch_ns)"
      CACHE_LOCK_WAIT_NS=0
      if [[ "$lock_end_ns" =~ ^[0-9]+$ ]] && [[ "$lock_start_ns" =~ ^[0-9]+$ ]] && [[ "$lock_end_ns" -ge "$lock_start_ns" ]]; then
        CACHE_LOCK_WAIT_NS=$((lock_end_ns - lock_start_ns))
      fi
      CACHE_LOCK_CONTENDED="$local_contended"
      return 0
    fi

    local_contended=1
    if [[ -d "$lock_dir" ]]; then
      now_sec="$(date +%s 2>/dev/null || printf "0")"
      lock_mtime="$(file_mtime_epoch "$lock_dir")"
      lock_age=0
      if [[ "$now_sec" =~ ^[0-9]+$ ]] && [[ "$lock_mtime" =~ ^[0-9]+$ ]] && [[ "$now_sec" -ge "$lock_mtime" ]]; then
        lock_age=$((now_sec - lock_mtime))
      fi
      if [[ "$lock_age" -ge "$CACHE_LOCK_STALE_SECONDS" ]]; then
        rm -f "${lock_dir}/pid" 2>/dev/null || true
        rmdir "$lock_dir" 2>/dev/null || true
        continue
      fi
    fi

    sleep "$CACHE_LOCK_POLL_SECONDS"
  done
}

yosys_resolved="$(command -v "$YOSYS_BIN" 2>/dev/null || printf "%s" "$YOSYS_BIN")"
if [[ -n "$MODES_CSV" ]]; then
  IFS=',' read -r -a modes_from_csv <<< "$MODES_CSV"
  for mode in "${modes_from_csv[@]}"; do
    mode="${mode#"${mode%%[![:space:]]*}"}"
    mode="${mode%"${mode##*[![:space:]]}"}"
    [[ -z "$mode" ]] && continue
    MODE_LIST+=("$mode")
  done
fi
if [[ -n "$MODE_COUNTS_CSV" ]]; then
  IFS=',' read -r -a mode_counts_from_csv <<< "$MODE_COUNTS_CSV"
  for mode_count in "${mode_counts_from_csv[@]}"; do
    mode_count="${mode_count#"${mode_count%%[![:space:]]*}"}"
    mode_count="${mode_count%"${mode_count##*[![:space:]]}"}"
    [[ -z "$mode_count" ]] && continue
    MODE_COUNT_LIST+=("$mode_count")
  done
fi
if [[ -n "$MODE_WEIGHTS_CSV" ]]; then
  IFS=',' read -r -a mode_weights_from_csv <<< "$MODE_WEIGHTS_CSV"
  for mode_weight in "${mode_weights_from_csv[@]}"; do
    mode_weight="${mode_weight#"${mode_weight%%[![:space:]]*}"}"
    mode_weight="${mode_weight%"${mode_weight##*[![:space:]]}"}"
    [[ -z "$mode_weight" ]] && continue
    MODE_WEIGHT_LIST+=("$mode_weight")
  done
fi
if [[ -n "$PROFILES_CSV" ]]; then
  IFS=',' read -r -a profiles_from_csv <<< "$PROFILES_CSV"
  for profile in "${profiles_from_csv[@]}"; do
    profile="${profile#"${profile%%[![:space:]]*}"}"
    profile="${profile%"${profile##*[![:space:]]}"}"
    [[ -z "$profile" ]] && continue
    PROFILE_LIST+=("$profile")
  done
fi
if [[ -n "$CFGS_CSV" ]]; then
  IFS=',' read -r -a cfgs_from_csv <<< "$CFGS_CSV"
  for cfg in "${cfgs_from_csv[@]}"; do
    cfg="${cfg#"${cfg%%[![:space:]]*}"}"
    cfg="${cfg%"${cfg##*[![:space:]]}"}"
    [[ -z "$cfg" ]] && continue
    CFG_LIST+=("$cfg")
  done
fi
if [[ -n "$SELECTS_CSV" ]]; then
  IFS=',' read -r -a selects_from_csv <<< "$SELECTS_CSV"
  for sel in "${selects_from_csv[@]}"; do
    sel="${sel#"${sel%%[![:space:]]*}"}"
    sel="${sel%"${sel##*[![:space:]]}"}"
    [[ -z "$sel" ]] && continue
    SELECT_LIST+=("$sel")
  done
fi
declare -a PROFILE_MODE_LIST=()
declare -a PROFILE_CFG_LIST=()
declare -a PROFILE_SELECT_LIST=()

mode_family_targets() {
  local mode_name="$1"
  case "$mode_name" in
    arith)
      printf "%s\n" "inv" "const0" "const1"
      ;;
    control)
      printf "%s\n" "cnot0" "cnot1"
      ;;
    stuck)
      printf "%s\n" "const0" "const1"
      ;;
    invert)
      printf "%s\n" "inv"
      ;;
    connect)
      printf "%s\n" "cnot0" "cnot1"
      ;;
    balanced|all)
      printf "%s\n" "inv" "const0" "const1" "cnot0" "cnot1"
      ;;
    *)
      printf "%s\n" "$mode_name"
      ;;
  esac
}

append_profile() {
  local profile_name="$1"
  case "$profile_name" in
    arith-depth)
      PROFILE_MODE_LIST+=("arith")
      PROFILE_CFG_LIST+=(
        "weight_pq_w=5"
        "weight_pq_mw=5"
        "weight_pq_b=3"
        "weight_pq_mb=3"
      )
      ;;
    control-depth)
      PROFILE_MODE_LIST+=("control")
      PROFILE_CFG_LIST+=(
        "weight_pq_c=5"
        "weight_pq_mc=5"
        "weight_pq_s=3"
        "weight_pq_ms=3"
      )
      ;;
    balanced-depth)
      PROFILE_MODE_LIST+=("arith" "control")
      PROFILE_CFG_LIST+=(
        "weight_pq_w=4"
        "weight_pq_mw=4"
        "weight_pq_c=4"
        "weight_pq_mc=4"
        "weight_pq_s=2"
        "weight_pq_ms=2"
      )
      ;;
    fault-basic)
      PROFILE_MODE_LIST+=("stuck" "invert" "connect")
      PROFILE_CFG_LIST+=(
        "weight_cover=5"
        "pick_cover_prcnt=80"
      )
      ;;
    fault-stuck)
      PROFILE_MODE_LIST+=("stuck" "invert")
      PROFILE_CFG_LIST+=(
        "weight_cover=4"
        "pick_cover_prcnt=70"
      )
      ;;
    fault-connect)
      PROFILE_MODE_LIST+=("connect" "invert")
      PROFILE_CFG_LIST+=(
        "weight_cover=4"
        "pick_cover_prcnt=70"
      )
      ;;
    cover)
      PROFILE_CFG_LIST+=(
        "weight_cover=5"
        "pick_cover_prcnt=80"
      )
      ;;
    none)
      ;;
    *)
      echo "Unknown --profile value: $profile_name (expected arith-depth|control-depth|balanced-depth|fault-basic|fault-stuck|fault-connect|cover|none)." >&2
      exit 1
      ;;
  esac
}

for profile in "${PROFILE_LIST[@]}"; do
  append_profile "$profile"
done

declare -a COMBINED_MODE_LIST=()
declare -A MODE_SEEN=()
declare -a FINAL_MODE_LIST=()
COMBINED_MODE_LIST=("${PROFILE_MODE_LIST[@]}" "${MODE_LIST[@]}")
declare -a MODE_COUNT_KEYS=()
declare -A MODE_COUNT_BY_MODE=()
mode_counts_enabled=0
mode_counts_total=0
declare -a MODE_WEIGHT_KEYS=()
declare -A MODE_WEIGHT_BY_MODE=()
mode_weights_enabled=0
mode_weights_total=0
for mode_count in "${MODE_COUNT_LIST[@]}"; do
  mode_name="${mode_count%%=*}"
  mode_value="${mode_count#*=}"
  if [[ -z "$mode_name" || "$mode_value" == "$mode_count" ]]; then
    echo "Invalid --mode-count entry: $mode_count (expected NAME=COUNT)." >&2
    exit 1
  fi
  mode_name="${mode_name#"${mode_name%%[![:space:]]*}"}"
  mode_name="${mode_name%"${mode_name##*[![:space:]]}"}"
  mode_value="${mode_value#"${mode_value%%[![:space:]]*}"}"
  mode_value="${mode_value%"${mode_value##*[![:space:]]}"}"
  if [[ -z "$mode_name" ]]; then
    echo "Invalid --mode-count entry: $mode_count (empty mode name)." >&2
    exit 1
  fi
  if [[ ! "$mode_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --mode-count count for $mode_name: $mode_value (expected positive integer)." >&2
    exit 1
  fi
  if [[ -z "${MODE_COUNT_BY_MODE[$mode_name]+x}" ]]; then
    MODE_COUNT_KEYS+=("$mode_name")
    MODE_COUNT_BY_MODE["$mode_name"]="$mode_value"
  else
    MODE_COUNT_BY_MODE["$mode_name"]="$((MODE_COUNT_BY_MODE[$mode_name] + mode_value))"
  fi
  mode_counts_total=$((mode_counts_total + mode_value))
  mode_counts_enabled=1
done
if [[ "$mode_counts_enabled" -eq 1 && "$mode_counts_total" -ne "$COUNT" ]]; then
  echo "Mode-count total ($mode_counts_total) must match --count ($COUNT)." >&2
  exit 1
fi
for mode_weight in "${MODE_WEIGHT_LIST[@]}"; do
  mode_name="${mode_weight%%=*}"
  mode_value="${mode_weight#*=}"
  if [[ -z "$mode_name" || "$mode_value" == "$mode_weight" ]]; then
    echo "Invalid --mode-weight entry: $mode_weight (expected NAME=WEIGHT)." >&2
    exit 1
  fi
  mode_name="${mode_name#"${mode_name%%[![:space:]]*}"}"
  mode_name="${mode_name%"${mode_name##*[![:space:]]}"}"
  mode_value="${mode_value#"${mode_value%%[![:space:]]*}"}"
  mode_value="${mode_value%"${mode_value##*[![:space:]]}"}"
  if [[ -z "$mode_name" ]]; then
    echo "Invalid --mode-weight entry: $mode_weight (empty mode name)." >&2
    exit 1
  fi
  if [[ ! "$mode_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --mode-weight weight for $mode_name: $mode_value (expected positive integer)." >&2
    exit 1
  fi
  if [[ -z "${MODE_WEIGHT_BY_MODE[$mode_name]+x}" ]]; then
    MODE_WEIGHT_KEYS+=("$mode_name")
    MODE_WEIGHT_BY_MODE["$mode_name"]="$mode_value"
  else
    MODE_WEIGHT_BY_MODE["$mode_name"]="$((MODE_WEIGHT_BY_MODE[$mode_name] + mode_value))"
  fi
  mode_weights_total=$((mode_weights_total + mode_value))
  mode_weights_enabled=1
done
if [[ "$mode_counts_enabled" -eq 1 && "$mode_weights_enabled" -eq 1 ]]; then
  echo "Use either --mode-count(s) or --mode-weight(s), not both." >&2
  exit 1
fi
if [[ "$mode_weights_enabled" -eq 1 ]]; then
  if [[ "$mode_weights_total" -le 0 ]]; then
    echo "Mode-weight total must be positive." >&2
    exit 1
  fi
  mode_counts_enabled=1
  mode_counts_total=0
  MODE_COUNT_KEYS=("${MODE_WEIGHT_KEYS[@]}")
  for mode_name in "${MODE_WEIGHT_KEYS[@]}"; do
    mode_value="${MODE_WEIGHT_BY_MODE[$mode_name]}"
    mode_count_value=$((COUNT * mode_value / mode_weights_total))
    MODE_COUNT_BY_MODE["$mode_name"]="$mode_count_value"
    mode_counts_total=$((mode_counts_total + mode_count_value))
  done
  mode_remainder=$((COUNT - mode_counts_total))
  mode_weight_key_count="${#MODE_WEIGHT_KEYS[@]}"
  if [[ "$mode_remainder" -gt 0 && "$mode_weight_key_count" -gt 0 ]]; then
    mode_remainder_start=$((SEED % mode_weight_key_count))
    for ((i=0; i<mode_remainder; ++i)); do
      key_idx=$(( (mode_remainder_start + i) % mode_weight_key_count ))
      mode_name="${MODE_WEIGHT_KEYS[$key_idx]}"
      MODE_COUNT_BY_MODE["$mode_name"]="$((MODE_COUNT_BY_MODE[$mode_name] + 1))"
    done
  fi
fi
if [[ "$mode_counts_enabled" -eq 1 ]]; then
  COMBINED_MODE_LIST+=("${MODE_COUNT_KEYS[@]}")
fi
for mode in "${COMBINED_MODE_LIST[@]}"; do
  mode="${mode#"${mode%%[![:space:]]*}"}"
  mode="${mode%"${mode##*[![:space:]]}"}"
  [[ -z "$mode" ]] && continue
  if [[ -n "${MODE_SEEN[$mode]+x}" ]]; then
    continue
  fi
  MODE_SEEN["$mode"]=1
  FINAL_MODE_LIST+=("$mode")
done
if [[ "${#FINAL_MODE_LIST[@]}" -eq 0 ]]; then
  FINAL_MODE_LIST+=("")
fi
MODE_LIST=("${FINAL_MODE_LIST[@]}")

declare -a COMBINED_CFG_LIST=()
declare -A CFG_BY_KEY=()
declare -a CFG_KEY_ORDER=()
COMBINED_CFG_LIST=("${PROFILE_CFG_LIST[@]}" "${CFG_LIST[@]}")
for cfg in "${COMBINED_CFG_LIST[@]}"; do
  key="${cfg%%=*}"
  value="${cfg#*=}"
  if [[ -z "$key" || "$value" == "$cfg" ]]; then
    echo "Invalid --cfg entry: $cfg (expected KEY=VALUE)." >&2
    exit 1
  fi
  if [[ ! "$key" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid --cfg key: $key (expected [A-Za-z0-9_]+)." >&2
    exit 1
  fi
  if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid --cfg value for $key: $value (expected integer)." >&2
    exit 1
  fi
  if [[ -z "${CFG_BY_KEY[$key]+x}" ]]; then
    CFG_KEY_ORDER+=("$key")
  fi
  CFG_BY_KEY["$key"]="$value"
done
CFG_LIST=()
for key in "${CFG_KEY_ORDER[@]}"; do
  CFG_LIST+=("${key}=${CFG_BY_KEY[$key]}")
done

declare -a COMBINED_SELECT_LIST=()
declare -A SELECT_SEEN=()
declare -a FINAL_SELECT_LIST=()
COMBINED_SELECT_LIST=("${PROFILE_SELECT_LIST[@]}" "${SELECT_LIST[@]}")
for sel in "${COMBINED_SELECT_LIST[@]}"; do
  sel="${sel#"${sel%%[![:space:]]*}"}"
  sel="${sel%"${sel##*[![:space:]]}"}"
  [[ -z "$sel" ]] && continue
  if [[ -n "${SELECT_SEEN[$sel]+x}" ]]; then
    continue
  fi
  SELECT_SEEN["$sel"]=1
  FINAL_SELECT_LIST+=("$sel")
done
SELECT_LIST=("${FINAL_SELECT_LIST[@]}")

cache_key=""
cache_file=""
if [[ -n "$CACHE_DIR" ]]; then
  design_hash="$(hash_file "$DESIGN")"
  mode_payload="$(printf "%s\n" "${MODE_LIST[@]}")"
  mode_count_payload="$(printf "%s\n" "${MODE_COUNT_LIST[@]}")"
  mode_weight_payload="$(printf "%s\n" "${MODE_WEIGHT_LIST[@]}")"
  profile_payload="$(printf "%s\n" "${PROFILE_LIST[@]}")"
  cfg_payload="$(printf "%s\n" "${CFG_LIST[@]}")"
  select_payload="$(printf "%s\n" "${SELECT_LIST[@]}")"
  cache_payload="$(
    cat <<EOF
v1
design_hash=$design_hash
top=$TOP
count=$COUNT
seed=$SEED
yosys_bin=$yosys_resolved
modes=$mode_payload
mode_counts=$mode_count_payload
mode_weights=$mode_weight_payload
profiles=$profile_payload
cfg=$cfg_payload
select=$select_payload
EOF
  )"
  cache_key="$(hash_string "$cache_payload")"
  cache_file="${CACHE_DIR}/${cache_key}.mutations.txt"
  CACHE_LOCK_WAIT_NS=0
  CACHE_LOCK_CONTENDED=0
  if [[ -s "$cache_file" ]]; then
    cache_saved_runtime_ns=0
    cache_meta_file="${cache_file}.meta"
    if [[ -f "$cache_meta_file" ]]; then
      cache_saved_runtime_ns="$(awk -F$'\t' '$1=="generation_runtime_ns"{print $2}' "$cache_meta_file" | head -n1)"
      cache_saved_runtime_ns="${cache_saved_runtime_ns:-0}"
      if [[ ! "$cache_saved_runtime_ns" =~ ^[0-9]+$ ]]; then
        cache_saved_runtime_ns=0
      fi
    fi
    cp "$cache_file" "$OUT_FILE"
    echo "Generated mutations: $(wc -l < "$OUT_FILE") (cache hit)"
    echo "Mutation file: $OUT_FILE"
    echo "Mutation generation runtime_ns: $(elapsed_ns_since "$SCRIPT_START_NS")"
    echo "Mutation cache saved_runtime_ns: $cache_saved_runtime_ns"
    echo "Mutation cache lock_wait_ns: $CACHE_LOCK_WAIT_NS"
    echo "Mutation cache lock_contended: $CACHE_LOCK_CONTENDED"
    echo "Mutation cache status: hit"
    echo "Mutation cache file: $cache_file"
    exit 0
  fi
  acquire_cache_lock "$cache_file"
  if [[ -s "$cache_file" ]]; then
    cache_saved_runtime_ns=0
    cache_meta_file="${cache_file}.meta"
    if [[ -f "$cache_meta_file" ]]; then
      cache_saved_runtime_ns="$(awk -F$'\t' '$1=="generation_runtime_ns"{print $2}' "$cache_meta_file" | head -n1)"
      cache_saved_runtime_ns="${cache_saved_runtime_ns:-0}"
      if [[ ! "$cache_saved_runtime_ns" =~ ^[0-9]+$ ]]; then
        cache_saved_runtime_ns=0
      fi
    fi
    cp "$cache_file" "$OUT_FILE"
    release_cache_lock
    echo "Generated mutations: $(wc -l < "$OUT_FILE") (cache hit)"
    echo "Mutation file: $OUT_FILE"
    echo "Mutation generation runtime_ns: $(elapsed_ns_since "$SCRIPT_START_NS")"
    echo "Mutation cache saved_runtime_ns: $cache_saved_runtime_ns"
    echo "Mutation cache lock_wait_ns: $CACHE_LOCK_WAIT_NS"
    echo "Mutation cache lock_contended: $CACHE_LOCK_CONTENDED"
    echo "Mutation cache status: hit"
    echo "Mutation cache file: $cache_file"
    exit 0
  fi
fi


mkdir -p "$(dirname "$OUT_FILE")"
WORK_DIR="$(mktemp -d)"

case "$DESIGN" in
  *.il) read_cmd="read_rtlil \"$DESIGN\"" ;;
  *.sv) read_cmd="read_verilog -sv \"$DESIGN\"" ;;
  *.v)  read_cmd="read_verilog \"$DESIGN\"" ;;
  *)
    echo "Unsupported design extension for $DESIGN (expected .il/.v/.sv)." >&2
    exit 1
    ;;
esac

prep_cmd="prep"
if [[ -n "$TOP" ]]; then
  prep_cmd="prep -top $TOP"
fi

mode_count="${#MODE_LIST[@]}"
base_count=0
extra_count=0
mode_extra_start=0
if [[ "$mode_counts_enabled" -eq 0 ]]; then
  base_count=$((COUNT / mode_count))
  extra_count=$((COUNT % mode_count))
  if [[ "$mode_count" -gt 0 ]]; then
    mode_extra_start=$((SEED % mode_count))
  fi
fi

declare -a MODE_OUT_FILES
declare -a MODE_TARGET_LIST=()
declare -a MODE_TARGET_COUNTS=()
declare -a ROUND_COUNTS=()

is_rotated_extra_index() {
  local index="$1"
  local start="$2"
  local extra="$3"
  local total="$4"
  local dist=0
  if [[ "$extra" -le 0 || "$total" -le 0 ]]; then
    return 1
  fi
  if [[ "$extra" -ge "$total" ]]; then
    return 0
  fi
  dist=$(( (index - start + total) % total ))
  if [[ "$dist" -lt "$extra" ]]; then
    return 0
  fi
  return 1
}

for idx in "${!MODE_LIST[@]}"; do
  mode="${MODE_LIST[$idx]}"
  list_count=0
  if [[ "$mode_counts_enabled" -eq 1 ]]; then
    if [[ -n "${MODE_COUNT_BY_MODE[$mode]+x}" ]]; then
      list_count="${MODE_COUNT_BY_MODE[$mode]}"
    fi
  else
    list_count="$base_count"
    if is_rotated_extra_index "$idx" "$mode_extra_start" "$extra_count" "$mode_count"; then
      list_count=$((list_count + 1))
    fi
  fi
  [[ "$list_count" -le 0 ]] && continue

  mapfile -t family_targets < <(mode_family_targets "$mode")
  family_count="${#family_targets[@]}"
  family_base=$((list_count / family_count))
  family_extra=$((list_count % family_count))
  family_extra_start=0
  if [[ "$family_count" -gt 0 ]]; then
    family_extra_start=$(( (SEED + idx) % family_count ))
  fi
  for family_idx in "${!family_targets[@]}"; do
    family_mode="${family_targets[$family_idx]}"
    family_list_count="$family_base"
    if is_rotated_extra_index "$family_idx" "$family_extra_start" "$family_extra" "$family_count"; then
      family_list_count=$((family_list_count + 1))
    fi
    [[ "$family_list_count" -le 0 ]] && continue
    MODE_TARGET_LIST+=("$family_mode")
    MODE_TARGET_COUNTS+=("$family_list_count")
  done
done

run_generation_round() {
  local round_tag="$1"
  local round_seed="$2"
  local idx=0
  local mode=""
  local list_count=0
  local script_file=""
  local mode_out_file=""
  local mode_arg=""
  local log_file=""
  local sources_file=""
  local mutate_cmd=""
  local key=""
  local value=""
  local any_target=0

  script_file="${WORK_DIR}/mutate.${round_tag}.ys"
  log_file="${WORK_DIR}/mutate.${round_tag}.log"
  MODE_OUT_FILES=()
  {
    echo "$read_cmd"
    echo "$prep_cmd"
  } > "$script_file"

  for idx in "${!MODE_TARGET_LIST[@]}"; do
    mode="${MODE_TARGET_LIST[$idx]}"
    list_count="${ROUND_COUNTS[$idx]:-0}"
    [[ "$list_count" -le 0 ]] && continue
    any_target=1

    sources_file="${WORK_DIR}/sources.${round_tag}.${idx}.txt"
    mode_out_file="${WORK_DIR}/mutations.${round_tag}.${idx}.txt"
    MODE_OUT_FILES+=("$mode_out_file")

    mode_arg=""
    if [[ -n "$mode" ]]; then
      mode_arg="-mode $mode"
    fi

    mutate_cmd="mutate -list $list_count -seed $round_seed -none"
    for cfg in "${CFG_LIST[@]}"; do
      key="${cfg%%=*}"
      value="${cfg#*=}"
      mutate_cmd+=" -cfg $key $value"
    done
    if [[ -n "$mode_arg" ]]; then
      mutate_cmd+=" $mode_arg"
    fi
    mutate_cmd+=" -o \"$mode_out_file\" -s \"$sources_file\""
    for sel in "${SELECT_LIST[@]}"; do
      mutate_cmd+=" $sel"
    done
    echo "$mutate_cmd" >> "$script_file"
  done

  if [[ "$any_target" -eq 0 ]]; then
    return
  fi

  "$YOSYS_BIN" -ql "$log_file" "$script_file"
  for mode_out_file in "${MODE_OUT_FILES[@]}"; do
    if [[ ! -f "$mode_out_file" ]]; then
      echo "Mutation generation failed: output file missing: $mode_out_file" >&2
      exit 1
    fi
  done
}

consume_generated_mutations() {
  local mode_out_file=""
  local raw_line=""
  local line=""
  local mut_id=""
  local mut_spec=""
  local done_flag=0

  for mode_out_file in "${MODE_OUT_FILES[@]}"; do
    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
      line="${raw_line#"${raw_line%%[![:space:]]*}"}"
      [[ -z "$line" ]] && continue
      [[ "${line:0:1}" == "#" ]] && continue
      mut_id="${line%%[[:space:]]*}"
      mut_spec="${line#"$mut_id"}"
      mut_spec="${mut_spec#"${mut_spec%%[![:space:]]*}"}"
      [[ -z "$mut_spec" ]] && continue
      if [[ -n "${SEEN_SPECS[$mut_spec]+x}" ]]; then
        continue
      fi
      SEEN_SPECS["$mut_spec"]=1
      printf "%s %s\n" "$next_id" "$mut_spec" >> "$OUT_FILE"
      next_id=$((next_id + 1))
      if [[ "$next_id" -gt "$COUNT" ]]; then
        done_flag=1
        break
      fi
    done < "$mode_out_file"
    if [[ "$done_flag" -eq 1 ]]; then
      break
    fi
  done
}

: > "$OUT_FILE"
declare -A SEEN_SPECS
next_id=1
MAX_TOPUP_ROUNDS=8

ROUND_COUNTS=("${MODE_TARGET_COUNTS[@]}")
run_generation_round "base" "$SEED"
consume_generated_mutations

for ((topup_round=1; topup_round<=MAX_TOPUP_ROUNDS; ++topup_round)); do
  [[ "$next_id" -gt "$COUNT" ]] && break
  needed=$((COUNT - next_id + 1))
  target_count="${#MODE_TARGET_LIST[@]}"
  topup_base=$((needed / target_count))
  topup_extra=$((needed % target_count))
  topup_extra_start=$(( (SEED + topup_round) % target_count ))
  ROUND_COUNTS=()
  for idx in "${!MODE_TARGET_LIST[@]}"; do
    topup_count="$topup_base"
    if is_rotated_extra_index "$idx" "$topup_extra_start" "$topup_extra" "$target_count"; then
      topup_count=$((topup_count + 1))
    fi
    ROUND_COUNTS+=("$topup_count")
  done
  run_generation_round "topup${topup_round}" "$((SEED + topup_round))"
  consume_generated_mutations
done

if [[ ! -s "$OUT_FILE" ]]; then
  echo "Mutation generation failed: generated mutation set is empty: $OUT_FILE" >&2
  exit 1
fi
if [[ "$next_id" -le "$COUNT" ]]; then
  generated_count=$((next_id - 1))
  echo "Mutation generation failed: unable to produce requested count after dedup/top-up (requested=$COUNT generated=$generated_count)." >&2
  exit 1
fi

if [[ -n "$cache_file" && -n "$cache_key" ]]; then
  cache_tmp="${cache_file}.tmp.$$"
  cache_meta_tmp="${cache_file}.meta.tmp.$$"
  cache_meta_file="${cache_file}.meta"
  cp "$OUT_FILE" "$cache_tmp"
  generation_runtime_ns="$(elapsed_ns_since "$SCRIPT_START_NS")"
  printf "generation_runtime_ns\t%s\n" "$generation_runtime_ns" > "$cache_meta_tmp"
  mv "$cache_tmp" "$cache_file"
  mv "$cache_meta_tmp" "$cache_meta_file"
  release_cache_lock
fi

generation_runtime_ns="${generation_runtime_ns:-$(elapsed_ns_since "$SCRIPT_START_NS")}"
echo "Generated mutations: $(wc -l < "$OUT_FILE")"
echo "Mutation file: $OUT_FILE"
echo "Mutation generation runtime_ns: $generation_runtime_ns"
echo "Mutation cache saved_runtime_ns: 0"
echo "Mutation cache lock_wait_ns: $CACHE_LOCK_WAIT_NS"
echo "Mutation cache lock_contended: $CACHE_LOCK_CONTENDED"
if [[ -n "$cache_file" ]]; then
  echo "Mutation cache status: miss"
  echo "Mutation cache file: $cache_file"
else
  echo "Mutation cache status: disabled"
fi
