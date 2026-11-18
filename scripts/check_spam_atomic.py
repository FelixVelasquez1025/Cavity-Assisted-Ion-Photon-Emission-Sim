"""Reproduce Stage 1a checks (not the Stage 1b benchmarks).

uv run python scripts/check_spam_atomic.py --output outputs/spam-stage1a.json
"""
import argparse
import json
from pathlib import Path
from caipe_sim.spam.validation import stage1a_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = json.dumps(stage1a_report(),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report+'\n')
    print(report)
