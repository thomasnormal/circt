#!/usr/bin/env python3
import argparse
import json
import sys


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
- Keep testing on ~/mbit/* for real world freed back.
- Remember to make unit tests as you go along and implement more features, and commit regularly.
- The work trees should regularly by merged into main, so the workers stay in sync.
Continue!
""".strip()


def main() -> int:
    payload = json.load(sys.stdin)

    with open("/Users/ahle/debug_stop_on_limits.txt", "a") as debug_file:
        print(f"stop_on_limits.py invoked with payload: {payload}", file=debug_file)

        transcript_path = payload.get("transcript_path")
        with open(transcript_path) as tf:
            lines = tf.readlines()
        last_msg = json.loads(lines[-1])
        print(f"last msg: {last_msg}", file=debug_file)

        error = last_msg.get("error")
        if error:
            print(f"Error detected ({error}); allowing stop.", file=debug_file)
            return 0

    sys.stderr.write(CONTINUE_MESSAGE)
    print(json.dumps({
        "decision": "block",
        "reason": CONTINUE_MESSAGE
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
