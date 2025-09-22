# app.py
import os
import sys
import traceback
import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import platform
import subprocess
from datetime import datetime, timezone
import streamlit as st
from zoneinfo import ZoneInfo

# Ensure local imports work when running "streamlit run app.py"
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

sys.path.append(REPO_ROOT)


def _repo_last_updated(repo_path: str) -> str:
    """Return a human-readable timestamp for the last git commit in ``repo_path``."""

    try:
        timestamp_raw = subprocess.check_output(
            ["git", "-C", repo_path, "log", "-1", "--format=%ct"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"

    if not timestamp_raw:
        return "unknown"

    try:
        timestamp = datetime.fromtimestamp(int(timestamp_raw), tz=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    except (ValueError, OSError, OverflowError):
        return "unknown"

    return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")

# --- Page setup ---
try:
    st.set_page_config(layout="wide")
except Exception:
    pass

st.markdown(
    (
        "<div style='display:flex; justify-content:flex-end; margin-bottom:0.5rem; color:#5c5c5c;'>"
        f"<span style='font-size:0.9rem;'>Last repository update: {_repo_last_updated(REPO_ROOT)}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.sidebar.title("Tools")

# Central registry: Display Name -> (module_path, callable_name)
tool_registry = {
    "Batch Convert": ("tools.batch_convert", "run"),
    "Brightness": ("tools.analyze_single_sif", "run"),
    "Saturation Series": ("tools.SaturationSeries", "run"),
    "Dye Colocalization": ("tools.colocalization", "run"),
    "Shelling Injection Table": ("tools.shelling_table", "run"),
    "Monomer Estimation": ("tools.monomers", "run"),
    "Process Movie": ("tools.read_movie", "run"),
    # "Plot CSVs": ("tools.plot_csv", "run"),  # commented like your original
    "TEM Size Analysis": ("tools.spherical_tem", "run"),
}

show_traces = st.sidebar.toggle(
    "Show error tracebacks",
    value=False,
    help="Expand to see full Python tracebacks when tools error."
)

# ---- Diagnostics Sidebar ----
with st.sidebar.expander("Diagnostics", expanded=False):
    # Basic environment
    st.markdown("**Environment**")
    st.code(
        f"python: {platform.python_version()}\n"
        f"os: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"executable: {sys.executable}\n"
        f"cwd: {os.getcwd()}",
        language="bash"
    )

    # Key library versions via importlib.metadata (avoids importing the libs)
    wanted_pkgs = [
        "numpy", "scipy", "scikit-image", "scikit-learn",
        "matplotlib", "pandas", "streamlit"
    ]
    st.markdown("**Package versions**")

    def pkg_version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "not installed"
        except Exception as e:
            return f"error: {type(e).__name__}"

    versions_text = "\n".join([f"{p}: {pkg_version(p)}" for p in wanted_pkgs])
    st.code(versions_text, language="bash")

    # sys.path (first few entries)
    st.markdown("**sys.path (first 5)**")
    st.code("\n".join(sys.path[:5]) + ("\n…",)[0], language="bash")

    # Tool availability without importing (fast + safe)
    st.markdown("**Tool availability (light check)**")
    rows = []
    for label, (modpath, funcname) in tool_registry.items():
        spec = importlib.util.find_spec(modpath)
        rows.append(f"{label}: module={'found' if spec else 'missing'}")
    st.code("\n".join(rows), language="bash")

    # Optional deep check for a specific tool (imports the module)
    st.markdown("**Deep check a tool** (imports module)")
    deep_tool = st.selectbox("Pick a tool to deep-check:", list(tool_registry.keys()))
    if st.button("Run deep check"):
        modpath, funcname = tool_registry[deep_tool]
        try:
            module = importlib.import_module(modpath)
            fn = getattr(module, funcname, None)
            callable_ok = callable(fn)
            st.success(
                f"Imported `{modpath}` OK. "
                f"{'Found' if fn else 'Missing'} `{funcname}`; "
                f"{'callable' if callable_ok else 'not callable'}."
            )
        except Exception as e:
            st.error(f"Deep check failed for {deep_tool}: {type(e).__name__}: {e}")
            if show_traces:
                st.code("".join(traceback.format_exception(e)), language="pytb")

# ---- Main tool launcher ----
available_labels = list(tool_registry.keys())
label_to_key = {label: key for label, key in tool_registry.items()}
tool_label = st.sidebar.radio("Analyze:", available_labels, index=0)

col1, col2 = st.columns([1, 2])

def render_error_context(title: str, err: Exception):
    st.error(f"⚠️ {title}: {type(err).__name__}: {err}")
    if show_traces:
        tb = "".join(traceback.format_exception(err))
        with st.expander("View traceback"):
            st.code(tb, language="pytb")

def safe_import(module_path: str):
    try:
        return importlib.import_module(module_path), None
    except Exception as e:
        return None, e

def safe_getattr(module, attr: str):
    try:
        fn = getattr(module, attr)
        if not callable(fn):
            raise TypeError(f"Attribute '{attr}' on '{module.__name__}' is not callable.")
        return fn, None
    except Exception as e:
        return None, e

def safe_run_tool(modpath: str, funcname: str, label: str):
    with st.spinner(f"Loading {label}…"):
        module, import_err = safe_import(modpath)
        if import_err:
            render_error_context(f"Failed to import {label} ({modpath})", import_err)
            return

    run_fn, getattr_err = safe_getattr(module, funcname)
    if getattr_err:
        render_error_context(f"Failed to find '{funcname}()' in {modpath}", getattr_err)
        return

    with st.spinner(f"Running {label}…"):
        try:
            return run_fn()
        except Exception as e:
            render_error_context(f"{label} crashed while running", e)
            return

if tool_label in label_to_key:
    modpath, funcname = label_to_key[tool_label]
    safe_run_tool(modpath, funcname, tool_label)
else:
    st.info("Select a tool from the sidebar to begin.")
