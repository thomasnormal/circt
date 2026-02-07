#!/usr/bin/env python3
import argparse
import json
import sys
import time


# Stop hook semantics:
# - exit 2: block stopping and continue
# - exit 0: allow stopping
CONTINUE_MESSAGE = """
Great! please update the project plan and the changelog on the work done.
The goal remains to bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Advice:
- What limitations do we still have? What features should we build?
- Check the status of each Track/Workstream. What task should they each do next?
- We should have four agents running at any one time.
- Don't worry about BMC and LEC, the Codex agent is working on those.
- Keep testing on ~/mbit/*avip* and ~/sv-tests/ and ~/verilator-verification/ and ~/yosys/tests/ and ~/opentitan/ for real world feedback.
- Make unit tests for every bug you fix or new feature you implement
- Keep lots of notes as you work, and a table of what features are still missing
- Commit regularly and merge with upstream main, so the workers stay in sync.
Continue!
""".strip()


def main() -> int:
    global CONTINUE_MESSAGE
    payload = json.load(sys.stdin)

    with open("/home/thomas-ahle/debug_stop_on_limits.txt", "a") as debug_file:
        print(f"stop_on_limits.py invoked with payload: {payload}", file=debug_file)

        transcript_path = payload.get("transcript_path")
        with open(transcript_path) as tf:
            lines = tf.readlines()

        for line in lines[-10::-1]:
            last_msg = json.loads(line)
            if last_msg.get("type") == "assistant":
                break
        else:
            raise Exception("Unable to find last assistant message. Stop" + "\n".join(lines[-10::-1]))
            return 0
        print(f"last msg: {last_msg}", file=debug_file)

        error = last_msg.get("error")
        if error:
            print(f"Error detected ({error}); allowing stop.", file=debug_file)
            return 0  # Allow stop on error instead of sleeping

        # last msg: {'type': 'queue-operation', 'operation': 'enqueue', 'timestamp': '2026-01-17T12:34:54.040Z', 'sessionId': '6785cec7-3fb3-48b9-8758-87cb4f0a07c1', 'content': '<task-notification>\n<task-id>a4b5a02</task-id>\n<status>completed</status>\n<summary>Agent "Track A: APB AVIP full compilation" completed</summary>\n<result>You\'ve hit your limit · resets 2pm (UTC)</result>\n</task-notification>\nFull transcript available at: /tmp/claude/-home-thomas-ahle-circt/tasks/a4b5a02.output'}
        content = last_msg.get("content")
        if content and "hit your limit" in content:
            print(f"Limit detected ({content}); allowing stop.", file=debug_file)
            return 0  # Allow stop on rate limit instead of sleeping

        # for i in range(-10,0):
        #     CONTINUE_MESSAGE += "\n" + lines[i]

    sys.stderr.write(CONTINUE_MESSAGE)
    print(json.dumps({
        "decision": "block",
        "reason": CONTINUE_MESSAGE
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
