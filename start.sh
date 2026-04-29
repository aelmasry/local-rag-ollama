#!/usr/bin/env bash
# اختصار: نفس run_all.sh — شغّل من مجلد rag-project: ./start.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_all.sh" "$@"
