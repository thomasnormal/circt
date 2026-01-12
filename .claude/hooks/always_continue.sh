
#!/usr/bin/env bash
set -euo pipefail

# Optional escape hatch: if this file exists, allow Claude to stop normally.
if [[ -f ".claude/ALLOW_STOP" ]]; then
     exit 0
fi

# IMPORTANT: write to stderr so Claude Code captures it as the continuation message
cat >&2 <<'MSG'
Great! please update the project plan on the work done, and what is still lacking to bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Advice:
- What limitations do we still have? What features should we build?
- Check the status of each Track/Workstream. What task should they each do next?
- We should have four agents running at any one time.
- Keep testing on ~/mbit/* for real world freed back.
- Remember to make unit tests as you go along and implement more features, and commit regularly.
- The work trees should regularly by merged into main, so the workers stay in sync.
Continue!
MSG

# Exit 2 = "block stopping and continue"
exit 2

