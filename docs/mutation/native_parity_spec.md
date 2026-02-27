# Native Mutation Planner Parity Spec

This document defines the CIRCT-only native mutation planning behavior and its
intended mapping to Yosys `mutate -list` concepts.

## Goals

- Keep existing deterministic legacy planning behavior as the default.
- Provide an opt-in weighted planner that mirrors the Yosys model directionally:
  weighted queue selection + coverage pressure + seed reproducibility.
- Keep third-party tools optional; do not require Yosys for CIRCT-only mode.
- Keep unsupported CIRCT-only options explicit (`--select(s)`).

## CLI Surface (CIRCT-only `circt-mut generate`)

- `--mode NAME` / `--modes CSV`
  - accepted names: `inv|const0|const1|cnot0|cnot1|arith|control|balanced|all|stuck|invert|connect`
  - mode names map to native text-operator subsets (approximate Yosys-family mapping)
- `--mode-count NAME=COUNT` / `--mode-counts CSV`
  - accepted in CIRCT-only mode
  - total must match `--count`
- `--mode-weight NAME=WEIGHT` / `--mode-weights CSV`
  - accepted in CIRCT-only mode
  - converted to per-mode counts by integer proportion + seeded remainder rotation
  - mutually exclusive with `--mode-count(s)`
- `--profile NAME` / `--profiles CSV`
  - accepted in CIRCT-only mode
  - expands into mode and planner-cfg presets using the same profile names as native generate
- `--cfg planner_policy=legacy|weighted`
  - default: `legacy`
  - auto-promotes to `weighted` when weight/cover knobs are provided without an
    explicit `planner_policy`
- `--cfg pick_cover_prcnt=<int>`
  - valid range: `0..100`
- `--cfg weight_cover=<int>`
- `--cfg weight_pq_w=<int>`
- `--cfg weight_pq_b=<int>`
- `--cfg weight_pq_c=<int>`
- `--cfg weight_pq_s=<int>`
- `--cfg weight_pq_mw=<int>`
- `--cfg weight_pq_mb=<int>`
- `--cfg weight_pq_mc=<int>`
- `--cfg weight_pq_ms=<int>`

All weight values must be `>= 0`. For `planner_policy=weighted`, the sum of all
`weight_*` values must be positive.

## Policies

### Legacy policy

- Existing deterministic round-robin over ordered operation list.
- Site index rotation is seed-stable and deterministic.
- This is the compatibility baseline.
- If `--mode(s)` is provided, planning is restricted to mapped native op subsets.
- If `--mode-count(s)` / `--mode-weight(s)` is provided, CIRCT-only generation
  emits per-mode plans and concatenates them using deterministic seed offsets.

### Weighted policy

- Builds one candidate per concrete `(op, site-index)` mutation site.
- Ops with no concrete sites are skipped in weighted planning.
- If no weighted candidates exist for the design, planner falls back to legacy
  emission to preserve deterministic count/output behavior.
- Tracks novelty-like coverage counters over synthetic keys:
  - src-ish key (line-derived)
  - wire-ish key
  - wirebit-ish key
- Adds diversity pressure across semantic buckets:
  - fault family (`compare|xcompare|logic|constant|arithmetic|shift|cast|misc`)
  - operator kind
  - coarse statement context (`control|assignment|verification|expression`)
- Adds realism bias in scoring:
  - favors control and assignment contexts over verification-only contexts.
- Adds anti-dominance penalties:
  - gradually penalizes repeated family/op/context picks to keep early schedules
    semantically distinct.
- Selects candidates by weighted bucket choice across:
  - cover bucket
  - primary queues (`w,b,c,s`)
  - module queues (`mw,mb,mc,ms`)
- Primary queue keys are global-family approximations; module queues add module
  scoping on top of the same base keys.
- Within queue picks:
  - with probability `pick_cover_prcnt`, prefer highest coverage score;
  - otherwise random among available queue candidates.
- Cover-bucket picks prioritize least-covered source first, with novelty score as
  deterministic tie-break.
- Seeded RNG is deterministic and uses rejection sampling for bounded draws.
- Auto-selected when weighted knobs are set via profiles or explicit `--cfg`
  entries (unless `planner_policy` is explicitly set to `legacy`).
- Site detection is lexer-aware for structural validity: mutation tokens inside
  comments and string literals are ignored.
- Arithmetic site detection (`ADD_TO_SUB`, `SUB_TO_ADD`) intentionally excludes
  `[...]` range/index contexts to avoid mutating packed-width declarations such
  as `[W-1:0]`.
- Multiplication mutations (`MUL_TO_ADD`, `ADD_TO_MUL`) use binary-site
  detection and intentionally skip reduction/wildcard/assignment contexts
  (`(*)`, `**`, `*=`) and `[...]` range/index expressions.
- Division/multiplication confusion mutations (`DIV_TO_MUL`, `MUL_TO_DIV`) use
  binary-site detection and intentionally skip operator-assignment/comment-like
  contexts (`/=`, `//`) and `[...]` range/index expressions.
- Unary arithmetic sign mutations (`UNARY_MINUS_DROP`) target unary-minus sites
  and avoid binary/compound contexts (`a-b`, `--`, `->`).
- Shift site detection (`SHL_TO_SHR`, `SHR_TO_SHL`) excludes triple-shift and
  shift-assignment spellings (`<<<`, `>>>`, `<<=`, `>>=`).
- Signed right-shift mutations (`SHR_TO_ASHR`, `ASHR_TO_SHR`) distinguish
  logical and arithmetic right-shift spellings and exclude operator-assignment
  forms (`>>>=`).
- `XOR_TO_OR` uses binary-XOR detection only (skips reduction/XNOR/assign
  forms like `^a`, `^~`, `~^`, `^=`).
- Constant mutations (`CONST0_TO_1`, `CONST1_TO_0`) cover sized binary/decimal/
  hex 1-bit literals and unsized tick literals (`1'b*`, `1'd*`, `1'h*`, `'0`,
  `'1`).
- Logical mutations (`AND_TO_OR`, `OR_TO_AND`) target short-circuit boolean
  operators (`&&`, `||`).
- Bitwise mutations (`BAND_TO_BOR`, `BOR_TO_BAND`) target binary bitwise
  operators (`&`, `|`) and exclude reduction/operator-assignment spellings.
- Unary inversion mutations split logical and bitwise intent:
  - `UNARY_NOT_DROP` targets logical negation (`!expr`).
  - `UNARY_BNOT_DROP` targets bitwise negation (`~expr`) and excludes reduction
    forms (`~&`, `~|`, `~^`) and XNOR token contexts (`^~`).
- Cast mutations (`SIGNED_TO_UNSIGNED`, `UNSIGNED_TO_SIGNED`) target
  `$signed(...)`/`$unsigned(...)` calls with boundary checks and optional
  whitespace before `(`.
- Mutation application must use the same site-index contract (code-only spans),
  otherwise `NATIVE_<OP>@<n>` can target different textual occurrences.

## Yosys Mapping Notes

Native weighted planner maps Yosys concepts approximately:

- Yosys candidate metadata (`module/cell/port/wire/src`) is approximated from
  textual mutation site location and op family.
- Yosys queue families and weight knobs are preserved by name and role.
- Yosys-like coverage pressure is represented via novelty score counters.

The mapping is intentionally incremental: preserve reproducibility and tuning
knobs first, then tighten metadata fidelity over time.

## Determinism Contract

For fixed:

- design text,
- `count`,
- `seed`,
- planner policy,
- planner `--cfg` values,

the generated mutation list must be deterministic.

## Cache Contract

Generation cache key includes:

- design hash,
- count/seed,
- profile list,
- mode list,
- mode-count and mode-weight inputs,
- ordered operation list,
- planner policy,
- all planner weight and cover parameters.

This prevents stale cross-policy cache hits.

## Future Tightening

- Replace textual site metadata with richer semantic candidate metadata.
- Add candidate filtering fields analogous to Yosys (`module/cell/port/...`).
- Add explicit parity regression corpus comparing native and Yosys schedules
  under fixed seeds.
