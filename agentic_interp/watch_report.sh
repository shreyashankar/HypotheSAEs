#!/bin/bash
cd "$(dirname "$0")/.."
LAST=""
while true; do
  CUR=$(ls -la agentic_interp/artifacts/agent_runs_v5 agentic_interp/artifacts/agent_runs_strict agentic_interp/artifacts/gepa_runs_v5 \
               agentic_interp/artifacts_congress/agent_runs_v5 agentic_interp/artifacts_congress/agent_runs_strict agentic_interp/artifacts_congress/gepa_runs_v5 2>/dev/null | md5)
  if [ "$CUR" != "$LAST" ]; then
    python3 agentic_interp/16_build_transcript_viewer.py 2>&1 | tail -1
    python3 agentic_interp/07_make_report.py 2>&1 | tail -1
    LAST="$CUR"
  fi
  sleep 45
done
