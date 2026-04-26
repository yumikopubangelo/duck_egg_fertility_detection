#!/usr/bin/env bash
# =============================================================================
# run.sh  --  Duck Egg Fertility Detection
# Compatible: Python 3.8 - 3.14 | Linux / macOS / Windows (Git Bash, WSL)
# =============================================================================
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
MIN_MAJOR=3
MIN_MINOR=8
PROJECT_NAME="Duck Egg Fertility Detection"

if [ -t 1 ]; then
    RED="\033[0;31m" GREEN="\033[0;32m" YELLOW="\033[1;33m"
    BLUE="\033[0;34m" CYAN="\033[0;36m" NC="\033[0m"
else
    RED="" GREEN="" YELLOW="" BLUE="" CYAN="" NC=""
fi

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${BLUE}=== $* ===${NC}"; }

# ---------------------------------------------------------------------------
# Find Python >= MIN_MAJOR.MIN_MINOR
# ---------------------------------------------------------------------------
find_python() {
    local candidates=(
        python3.14 python3.13 python3.12 python3.11 python3.10
        python3.9  python3.8  python3    python
    )
    for py in "${candidates[@]}"; do
        if command -v "$py" &>/dev/null; then
            local ver major minor
            ver=$("$py" -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>/dev/null) || continue
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -gt "$MIN_MAJOR" ] || \
               { [ "$major" -eq "$MIN_MAJOR" ] && [ "$minor" -ge "$MIN_MINOR" ]; }; then
                echo "$py"
                return 0
            fi
        fi
    done
    error "No Python ${MIN_MAJOR}.${MIN_MINOR}+ found. Install from https://www.python.org/downloads/"
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        section "Creating virtual environment"
        local PYTHON
        PYTHON=$(find_python)
        info "Using: $PYTHON ($($PYTHON --version))"
        "$PYTHON" -m venv "$VENV_DIR"
        ok "Created $VENV_DIR/"
    fi
    if   [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV_DIR/Scripts/activate"    # Windows Git Bash
    elif [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"         # Linux / macOS / WSL
    else
        error "Cannot find venv activation script in $VENV_DIR."
    fi
    ok "Active: $(python --version)"
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cmd_setup() {
    section "Setup"
    ensure_venv
    info "Upgrading pip, setuptools, wheel ..."
    python -m pip install --upgrade pip setuptools wheel --quiet
    info "Installing core requirements ..."
    pip install -r requirements.txt
    info "Installing web requirements ..."
    pip install -r requirements-web.txt
    info "Installing project (editable + dev extras) ..."
    pip install -e ".[dev]"
    ok "Setup complete.  Run  ./run.sh help  for available commands."
}

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
cmd_preprocess() { section "Step 1  Preprocess";    ensure_venv; python scripts/01_preprocess_data.py   "$@"; }
cmd_annotate()   { section "Step 2  Annotate";      ensure_venv; python scripts/02_create_annotations.py "$@"; }
cmd_train()      { section "Step 3  Train U-Net";   ensure_venv; python scripts/03_train_unet.py         "$@"; }
cmd_masks()      { section "Step 3b Masks";          ensure_venv; python scripts/03b_generate_masks.py    "$@"; }
cmd_features()   { section "Step 4  Features";      ensure_venv; python scripts/04_extract_features.py   "$@"; }
cmd_train_awc()  { section "Step 5  Train AWC";     ensure_venv; python scripts/05_train_awc.py           "$@"; }
cmd_evaluate()   { section "Step 6  Evaluate";      ensure_venv; python scripts/06_evaluate_models.py     "$@"; }
cmd_predict()    { section "Step 7  Predict";       ensure_venv; python scripts/07_inference.py            "$@"; }
cmd_retrain()    { section "Step 8  Retrain";       ensure_venv; python scripts/08_retrain_model.py        "$@"; }

cmd_pipeline() {
    section "Full pipeline  (steps 1 -> 8)"
    cmd_preprocess "$@"
    cmd_annotate
    cmd_train
    cmd_masks
    cmd_features
    cmd_train_awc
    cmd_evaluate
    info "Pipeline complete. Run  ./run.sh predict  for inference."
}

# ---------------------------------------------------------------------------
# Web / App
# ---------------------------------------------------------------------------
cmd_web() {
    section "Flask API"
    ensure_venv
    local port="${PORT:-5000}"
    info "Listening on http://0.0.0.0:${port}"
    python -m flask --app web/api/app.py run --host 0.0.0.0 --port "$port" "$@"
}

cmd_streamlit() {
    section "Streamlit dashboard"
    ensure_venv
    local port="${PORT:-8501}"
    info "Opening http://localhost:${port}"
    streamlit run app/app.py --server.port "$port" "$@"
}

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
cmd_test()   { section "Tests";  ensure_venv; python -m pytest tests/ -v --cov=src/ --cov-report=term-missing "$@"; }
cmd_lint()   { section "Lint";   ensure_venv; ruff check . && black --check .; }
cmd_format() { section "Format"; ensure_venv; black . && ruff check --fix .; }

cmd_clean() {
    section "Clean"
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc"       -not -path "./.venv/*" -delete           2>/dev/null || true
    rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ 2>/dev/null || true
    ok "Clean done."
}

cmd_info() {
    section "Environment info"
    ensure_venv
    echo "Python  : $(python --version)"
    echo "pip     : $(pip --version)"
    echo "OS      : $(uname -s 2>/dev/null || echo Windows)"
    echo ""
    pip list 2>/dev/null | grep -Ei "torch|tensorflow|numpy|opencv|flask|streamlit" || true
}

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
cmd_help() {
    echo ""
    echo "${BLUE}${PROJECT_NAME}${NC}  --  run.sh"
    echo ""
    echo "Usage:  ./run.sh <command> [args]"
    echo ""
    echo "${CYAN}Setup${NC}"
    echo "  setup         Create .venv and install all dependencies"
    echo "  clean         Remove __pycache__, build artefacts"
    echo "  info          Show Python and package versions"
    echo ""
    echo "${CYAN}Pipeline${NC}"
    echo "  pipeline      Full training pipeline  (steps 1-8)"
    echo "  preprocess    Step 1  Preprocess raw images"
    echo "  annotate      Step 2  Create annotations"
    echo "  train         Step 3  Train U-Net model"
    echo "  masks         Step 3b Generate segmentation masks"
    echo "  features      Step 4  Extract features"
    echo "  train-awc     Step 5  Train AWC classifier"
    echo "  evaluate      Step 6  Evaluate all models"
    echo "  predict       Step 7  Run inference on new images"
    echo "  retrain       Step 8  Retrain with new data"
    echo ""
    echo "${CYAN}Web${NC}"
    echo "  web           Flask API          (default PORT=5000)"
    echo "  streamlit     Streamlit dashboard (default PORT=8501)"
    echo ""
    echo "${CYAN}Development${NC}"
    echo "  test          pytest with coverage"
    echo "  lint          ruff + black --check"
    echo "  format        black + ruff --fix"
    echo ""
    echo "${CYAN}Env vars${NC}"
    echo "  PORT=8080         ./run.sh web        override port"
    echo "  VENV_DIR=myenv    ./run.sh setup       override venv dir"
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
    setup)        cmd_setup       "$@" ;;
    clean)        cmd_clean              ;;
    info)         cmd_info               ;;
    pipeline)     cmd_pipeline    "$@" ;;
    preprocess)   cmd_preprocess  "$@" ;;
    annotate)     cmd_annotate    "$@" ;;
    train)        cmd_train       "$@" ;;
    masks)        cmd_masks       "$@" ;;
    features)     cmd_features    "$@" ;;
    train-awc)    cmd_train_awc   "$@" ;;
    evaluate)     cmd_evaluate    "$@" ;;
    predict)      cmd_predict     "$@" ;;
    retrain)      cmd_retrain     "$@" ;;
    web)          cmd_web         "$@" ;;
    streamlit)    cmd_streamlit   "$@" ;;
    test)         cmd_test        "$@" ;;
    lint)         cmd_lint               ;;
    format)       cmd_format             ;;
    help|--help|-h) cmd_help            ;;
    *)  error "Unknown command: ${CMD}. Run ./run.sh help" ;;
esac