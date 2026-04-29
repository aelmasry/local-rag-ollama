#!/usr/bin/env bash
# تشغيل/إيقاف نظام RAG المحلي: Ollama + FastAPI + Streamlit
# Usage: ./run_all.sh [start|stop|restart|status]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Default 8010: port 8000 is often Laravel/PHP (HTML 404, not this API).
BACKEND_PORT="${BACKEND_PORT:-8010}"
UI_PORT="${UI_PORT:-8501}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

log() {
  printf "[rag] %s\n" "$1"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status}

  start    — Ollama (if needed), models, FastAPI on :${BACKEND_PORT}, Streamlit on :${UI_PORT}
  stop     — stop this project's uvicorn + streamlit (does not stop Ollama)
  restart  — stop then start
  status   — quick health + ports

Env: BACKEND_PORT, UI_PORT, OLLAMA_BASE_URL
EOF
}

check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    exit 1
  fi
}

is_port_open() {
  lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

rag_health_ok() {
  local p="$1"
  curl -sS --max-time 2 "http://127.0.0.1:${p}/health" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'
}

stop_rag_processes() {
  log "Stopping RAG backend + UI (if running)..."
  pkill -f "uvicorn backend.main:app" 2>/dev/null && log "Stopped uvicorn (backend)." || true
  pkill -f "streamlit run ui/app.py" 2>/dev/null && log "Stopped streamlit (UI)." || true
  sleep 1
}

cmd_status() {
  log "Backend :${BACKEND_PORT}"
  if rag_health_ok "$BACKEND_PORT"; then
    curl -sS "http://127.0.0.1:${BACKEND_PORT}/health" | head -c 500
    echo ""
  else
    log "  (not healthy or not running)"
  fi
  log "UI :${UI_PORT} open? $(is_port_open "$UI_PORT" && echo yes || echo no)"
  log "Ollama: $(curl -sS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 && echo ok || echo down)"
}

start_ollama_if_needed() {
  if curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    log "Ollama is already running."
    return
  fi

  log "Starting Ollama server..."
  nohup ollama serve >/tmp/rag_ollama.log 2>&1 &
  sleep 2
  if ! curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    log "Ollama failed to start. Check /tmp/rag_ollama.log"
    exit 1
  fi
  log "Ollama started."
}

ensure_models() {
  local models
  models="$(ollama list 2>/dev/null || true)"
  if [[ "$models" != *"llama3"* ]]; then
    log "Pulling missing model: llama3"
    ollama pull llama3
  fi
  if [[ "$models" != *"nomic-embed-text"* ]]; then
    log "Pulling missing model: nomic-embed-text"
    ollama pull nomic-embed-text
  fi
}

start_backend() {
  if is_port_open "$BACKEND_PORT" && rag_health_ok "$BACKEND_PORT"; then
    log "RAG backend already running on :$BACKEND_PORT"
    return
  fi
  if is_port_open "$BACKEND_PORT" && ! rag_health_ok "$BACKEND_PORT"; then
    log "Port $BACKEND_PORT is in use but /health is not this API. Set BACKEND_PORT=8020 and retry."
    exit 1
  fi

  log "Starting FastAPI backend on :$BACKEND_PORT"
  nohup .venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
    >/tmp/rag_backend.log 2>&1 &
  sleep 2
  if ! rag_health_ok "$BACKEND_PORT"; then
    log "Backend failed to start. Check /tmp/rag_backend.log"
    exit 1
  fi
}

start_ui() {
  if is_port_open "$UI_PORT"; then
    if pgrep -fl "streamlit run ui/app.py" >/dev/null 2>&1; then
      log "Streamlit UI already running on :$UI_PORT"
      return
    fi
    log "Port $UI_PORT is in use (not our Streamlit?). Set UI_PORT=8502 or free the port."
    exit 1
  fi

  log "Starting Streamlit UI on :$UI_PORT"
  nohup env API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    .venv/bin/python -m streamlit run ui/app.py --server.port "$UI_PORT" --server.address 0.0.0.0 \
    >/tmp/rag_ui.log 2>&1 &
  sleep 3
  if ! is_port_open "$UI_PORT"; then
    log "UI failed to start. Check /tmp/rag_ui.log"
    exit 1
  fi
}

cmd_start() {
  check_cmd curl
  check_cmd lsof
  check_cmd ollama

  if [[ ! -x ".venv/bin/python" ]]; then
    log "No venv. Run:"
    log "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
  fi

  start_ollama_if_needed
  ensure_models
  start_backend
  start_ui

  log "Done."
  log "  API:  http://127.0.0.1:${BACKEND_PORT}/docs"
  log "  UI:   http://127.0.0.1:${UI_PORT}"
  log "  Logs: /tmp/rag_ollama.log | /tmp/rag_backend.log | /tmp/rag_ui.log"
}

cmd_restart() {
  stop_rag_processes
  cmd_start
}

ACTION="${1:-start}"
case "$ACTION" in
  start) cmd_start ;;
  stop) stop_rag_processes; log "Stopped. Ollama left running." ;;
  restart) cmd_restart ;;
  status) cmd_status ;;
  -h|--help|help) usage ;;
  *)
    log "Unknown: $ACTION"
    usage
    exit 1
    ;;
esac
