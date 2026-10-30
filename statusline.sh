#!/bin/sh
# Claude Code status line: shows vtype's recording state when Claude runs under vtype.
cat > /dev/null  # Claude Code sends session JSON on stdin; not needed here
[ -n "$VTYPE_STATE" ] && [ -f "$VTYPE_STATE" ] && cat "$VTYPE_STATE"
exit 0
