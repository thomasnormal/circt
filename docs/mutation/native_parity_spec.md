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
  - `cnot0`/`cnot1` are polarity-specific in CIRCT-only mode (`if`/mux control
    forced-low vs forced-high style mutations).
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
- Modulo/division confusion mutations (`MOD_TO_DIV`, `DIV_TO_MOD`) use the same
  binary-site detection guardrails and skip operator-assignment contexts
  (`%=`, `/=`) and `[...]` range/index expressions.
- Unary arithmetic sign mutations (`UNARY_MINUS_DROP`) target unary-minus sites
  and avoid binary/compound contexts (`a-b`, `--`, `->`).
- Compound-assignment arithmetic confusion mutations
  (`PLUS_EQ_TO_MINUS_EQ`, `MINUS_EQ_TO_PLUS_EQ`, `MUL_EQ_TO_DIV_EQ`,
  `DIV_EQ_TO_MUL_EQ`) target procedural compound-assignment tokens with
  declaration/context guards and deterministic site indexing.
- Compound-assignment shift-direction confusion mutations
  (`SHL_EQ_TO_SHR_EQ`, `SHR_EQ_TO_SHL_EQ`) target procedural shift-assignment
  tokens (`<<=`, `>>=`) and exclude arithmetic-shift assignment overlap
  (`<<<=`, `>>>=`) via token-boundary guards.
- Shift site detection (`SHL_TO_SHR`, `SHR_TO_SHL`) excludes triple-shift and
  shift-assignment spellings (`<<<`, `>>>`, `<<=`, `>>=`).
- Signed right-shift mutations (`SHR_TO_ASHR`, `ASHR_TO_SHR`) distinguish
  logical and arithmetic right-shift spellings and exclude operator-assignment
  forms (`>>>=`).
- `XOR_TO_OR` uses binary-XOR detection only (skips reduction/XNOR/assign
  forms like `^a`, `^~`, `~^`, `^=`).
- Relational polarity swaps (`LT_TO_GT`, `GT_TO_LT`, `LE_TO_GE`, `GE_TO_LE`)
  reuse standalone/relational comparator site detection to preserve structural
  validity and deterministic site-index contracts.
- XOR/XNOR confusion mutations:
  - `XOR_TO_XNOR` uses the same binary-XOR site detection as `XOR_TO_OR`.
  - `XNOR_TO_XOR` targets binary XNOR spellings (`^~`, `~^`) and skips
    reduction-XNOR forms.
- X-sensitivity compare swaps:
  - `EQ_TO_CASEEQ` and `NEQ_TO_CASENEQ` target only binary `==` / `!=` sites
    (never `===` / `!==`), preserving deterministic site-index contracts.
  - `CASEEQ_TO_EQ` and `CASENEQ_TO_NEQ` target only binary `===` / `!==`
    sites.
- Constant mutations (`CONST0_TO_1`, `CONST1_TO_0`) cover sized binary/decimal/
  hex 1-bit literals and unsized tick literals (`1'b*`, `1'd*`, `1'h*`, `'0`,
  `'1`).
- Logical mutations (`AND_TO_OR`, `OR_TO_AND`) target short-circuit boolean
  operators (`&&`, `||`).
- Logical/bitwise confusion mutations (`LAND_TO_BAND`, `LOR_TO_BOR`,
  `BAND_TO_LAND`, `BOR_TO_LOR`) model swapped boolean-vs-bitwise intent while
  preserving binary-operator structure.
- Bitwise mutations (`BAND_TO_BOR`, `BOR_TO_BAND`) target binary bitwise
  operators (`&`, `|`) and exclude reduction/operator-assignment spellings.
- Unary inversion mutations split logical and bitwise intent:
  - `UNARY_NOT_DROP` targets logical negation (`!expr`).
  - `UNARY_BNOT_DROP` targets bitwise negation (`~expr`) and excludes reduction
    forms (`~&`, `~|`, `~^`) and XNOR token contexts (`^~`).
- Assignment timing mutations model procedural update-order faults:
  - `BA_TO_NBA` targets procedural blocking assignments (`=`) and rewrites to
    nonblocking (`<=`) with declaration/comparator/continuous-assignment guards,
    including typed-declaration exclusion (`my_t v = ...`).
  - `NBA_TO_BA` targets procedural nonblocking assignments (`<=`) and rewrites
    to blocking (`=`) with matching site-index guards and the same typed-
    declaration exclusion.
- Ternary mux-arm mutations model swapped-data control bugs:
  - `MUX_SWAP_ARMS` targets assignment-context ternary expressions
    (`cond ? tval : fval`) and rewrites to (`cond ? fval : tval`) with
    statement-level disqualifier guards to avoid wildcard-case token confusion.
- Ternary mux control-stuck mutations model control-value forcing bugs:
  - `MUX_FORCE_TRUE` targets assignment-context ternary expressions and rewrites
    to an always-true-arm equivalent (`cond ? tval : tval`).
  - `MUX_FORCE_FALSE` targets assignment-context ternary expressions and
    rewrites to an always-false-arm equivalent (`cond ? fval : fval`).
- Structured control-arm swap mutations model decode/control wiring mistakes:
  - `IF_ELSE_SWAP_ARMS` swaps `if/else` branch bodies for `if (...) ... else ...`
    forms when both arms are structurally valid and unambiguous.
  - `CASE_ITEM_SWAP_ARMS` swaps adjacent `case` item bodies in a conservative
    subset (simple item bodies and balanced `begin/end` forms) to preserve
    structural validity and deterministic site-index contracts.
- Conditional-polarity mutations model inverted control intent:
  - `IF_COND_NEGATE` targets `if (cond)` headers and rewrites to
    `if (!(cond))` with word-boundary token matching and balanced-parenthesis
    guards.
  - `IF_COND_TRUE` and `IF_COND_FALSE` force `if` conditions to `1'b1` and
    `1'b0` respectively.
  - `RESET_COND_NEGATE` targets reset-like `if` conditions (`rst*` / `*reset*`)
    and inverts only those reset conditions.
  - `RESET_COND_TRUE` and `RESET_COND_FALSE` force only reset-like `if`
    conditions to `1'b1` / `1'b0`.
- Edge-polarity mutations model clock/reset sensitivity bugs:
  - `POSEDGE_TO_NEGEDGE` targets `posedge` event-control keywords with
    identifier-boundary guards.
  - `NEGEDGE_TO_POSEDGE` targets `negedge` event-control keywords with
    identifier-boundary guards.
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
