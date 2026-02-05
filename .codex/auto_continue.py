#!/usr/bin/env python3
import json, os, re, subprocess, sys, time

MSG = """
please continue

Some general advice:
- Follow the project plan, and update the changelog regularly.
- What limitations do we still have? What features should we build?
- Don't shy away from ambitious and complicated tasks. Focus on what we need long term.
- Test on ~/mbit/*avip* and ~/sv-tests/ and ~/verilator-verification/ and ~/yosys/tests/ and ~/opentitan/ regularly.
- Make unit tests for every bug you fix or new feature you implement
- Commit regularly and merge with upstream main.
- Don't be worried if some files are changed without your knowledge. It might just be other agents working on the project.
- If choosing between multiple options, pick the one that's the best choice long term.

Continue!
"""

FATAL_RE = re.compile(
    r"""
    (insufficient[_ ]quota|out\s+of\s+credits|exceeded\s+your\s+current\s+quota|
     billing|payment\s+required|
     invalid[_ ]api[_ ]key|unauthorized|forbidden|authentication\s+failed|
     rate\s+limit|too\s+many\s+requests|\b429\b|\b403\b|\b401\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass

def load_payload() -> dict:
    if len(sys.argv) > 1 and sys.argv[1]:
        return json.loads(sys.argv[1])
    raw = sys.stdin.read()
    if raw:
        return json.loads(raw)
    return {}

payload = load_payload()
cwd = payload.get("cwd") or os.getcwd()

def write_debug(payload: dict, cwd: str) -> None:
    try:
        os.makedirs(os.path.join(cwd, ".codex"), exist_ok=True)
        # NOTE: Avoid writing to tracked files by default. The `auto_continue`
        # runner frequently updates debug state, which is useful locally but
        # should not show up in `git status` during development.
        #
        # Use CODEX_AUTO_CONTINUE_WRITE_TRACKED_DEBUG=1 to restore the previous
        # behavior of writing to `.codex/auto_continue.last.json`.
        debug_path = os.path.join(cwd, ".codex", "auto_continue.last.local.json")
        if os.environ.get("CODEX_AUTO_CONTINUE_WRITE_TRACKED_DEBUG", "") == "1":
            debug_path = os.path.join(cwd, ".codex", "auto_continue.last.json")
        panes = None
        try:
            panes_out = subprocess.run(
                ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_active} #{pane_current_command}"],
                check=False,
                capture_output=True,
                text=True,
            )
            if panes_out.stdout:
                panes = panes_out.stdout.strip().splitlines()
        except Exception:
            panes = None
        debug = {
            "payload": payload,
            "payload_type": payload.get("type"),
            "tmux_pane": os.environ.get("TMUX_PANE"),
            "codex_tmux_pane": os.environ.get("CODEX_TMUX_PANE"),
            "tmux_panes": panes,
        }
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, sort_keys=True)
    except Exception:
        pass

write_debug(payload, cwd)

payload_type = payload.get("type", "")
# notify currently fires on turn completion events (commonly: agent-turn-complete)
if payload_type and "turn-complete" not in payload_type:
    sys.exit(0)

last = payload.get("last-assistant-message") or ""

# Per-project pause file (lives in the repo’s .codex/)
pause_file = os.path.join(cwd, ".codex", "AUTO_CONTINUE_PAUSE")

# If we hit a likely “credits/quota/auth/rate limit” situation, stop auto-continue.
if FATAL_RE.search(last):
    touch(pause_file)

    pane = os.environ.get("TMUX_PANE")
    if pane:
        subprocess.run(
            ["tmux", "display-message", "-t", pane, "Codex auto-continue paused (quota/auth/rate-limit suspected)."],
            check=False,
        )
    sys.exit(0)

# Manual pause
if os.path.exists(pause_file):
    sys.exit(0)

# Inject the next prompt into the same tmux pane Codex is running in.
pane = os.environ.get("CODEX_TMUX_PANE") or os.environ.get("TMUX_PANE")
if not pane:
    sys.stderr.write("auto_continue: TMUX_PANE not set; skipping send-keys\n")
    sys.exit(0)

time.sleep(0.2)
subprocess.run(
    ["tmux", "send-keys", "-t", pane, "-l", MSG],
    check=False,
)
subprocess.run(
    ["tmux", "send-keys", "-t", pane, "C-m"],
    check=False,
)
