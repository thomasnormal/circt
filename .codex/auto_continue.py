#!/usr/bin/env python3
import json, os, re, subprocess, sys, time
from datetime import datetime

# Note: You must run with Approvals: Full, to disable the sandbox, for this script to work

LOG_PATH = "/tmp/codex_auto_continue.log"

def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {msg.rstrip()}\n")

log(f"invoke argv_len={len(sys.argv)} tmux={os.environ.get('TMUX')} pane={os.environ.get('TMUX_PANE')}")
if len(sys.argv) < 2:
    log("exit: missing payload arg")
    sys.exit(0)

try:
    payload = json.loads(sys.argv[1])
except Exception as e:
    log(f"exit: invalid payload json: {e}")
    sys.exit(0)

# Only fire after a turn completes (currently the main supported notify event). :contentReference[oaicite:3]{index=3}
if payload.get("type") != "agent-turn-complete":
    log(f"exit: wrong event type={payload.get('type')}")
    sys.exit(0)

# Safety valve: touch this file to pause auto-continue.
# pause_file = os.path.expanduser("~/.codex/AUTO_CONTINUE_PAUSE")
# if os.path.exists(pause_file):
#     sys.exit(0)

# Optional: stop if the assistant clearly says it’s finished.
last = payload.get("last-assistant-message", "") or ""
# if re.search(r"\b(DONE|all done|task complete|completed everything)\b", last, re.I):
#     sys.exit(0)

# Stop if we hit context/token limits
token_limit_re = r"(token limit|context window|context length|maximum context|too many tokens)"
if re.search(token_limit_re, last, re.I):
    log("exit: last message hit token/context limit")
    sys.exit(0)

# This works best when Codex is running inside tmux (TMUX_PANE is set).
pane = os.environ.get("TMUX_PANE")
if not pane:
    log("exit: TMUX_PANE not set")
    sys.exit(0)

# Give the TUI a beat to settle before injecting keys.
time.sleep(0.2)

# Inject your “keep going” nudge + Enter.
# Tip: make this a stronger instruction than "continue".
msg = """
please continue

Some general advice:
- Follow the project plan, and update the changelog regularly.
- What limitations do we still have? What features should we build?
- Don't shy away from ambitious and complicated tasks. Focus on what we need long term.
- Test on ~/mbit/*avip* and ~/sv-tests/ and ~/verilator-verification/ regularly.
- Remember to make unit tests as you go along and implement more features, and commit regularly. Merge regularly with upstream main.
- Don't be worried if some files are changed without your knowledge. It might just be other agents working on the project.

Continue!
"""
# msg = "continue"
# msg = "Keep going. Take the next step. If you're blocked, ask ONE clear question."
result = subprocess.run(["tmux", "send-keys", "-t", pane, msg, "C-m"], check=False)
if result.returncode != 0:
    log(f"send-keys failed: rc={result.returncode}")
else:
    log("sent: continue message")
