"""
Superb Python REPL Tool
=======================
A persistent Python execution environment for the agent.
Uses a SUBPROCESS for true process isolation — can be cleanly killed on timeout.

PLOT CAPTURE: When running in web mode, plots are captured via callback.
"""

import sys
import io
import json
import logging
import gc
import os
import re
import base64
import tempfile
import subprocess
import threading
import traceback
import matplotlib
# Force non-interactive backend to prevent crashes on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Pre-import for custom colormaps

logger = logging.getLogger(__name__)
import matplotlib.cm as cm  # Pre-import for colormap access

# =============================================================================
# PUBLICATION-GRADE LIGHT THEME (white background for academic papers)
# =============================================================================
_EURUS_STYLE = {
    # ── Figure ──
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.dpi": 300,          # 300 DPI for print-quality
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    # ── Axes ──
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.titlecolor": "#000000",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    # ── Grid ──
    "grid.color": "#d0d0d0",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    # ── Ticks ──
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # ── Text ──
    "text.color": "#1a1a1a",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    # ── Lines ──
    "lines.linewidth": 1.8,
    "lines.antialiased": True,
    "lines.markersize": 5,
    # ── Legend ──
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.shadow": False,
    # ── Colorbar ──
    "image.cmap": "viridis",
    # ── Patches ──
    "patch.edgecolor": "#333333",
}
matplotlib.rcParams.update(_EURUS_STYLE)

# Curated color cycle for white backgrounds (high-contrast, publication-safe)
_EURUS_COLORS = [
    "#1f77b4",  # steel blue
    "#d62728",  # brick red
    "#2ca02c",  # forest green
    "#ff7f0e",  # orange
    "#9467bd",  # muted purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
]
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=_EURUS_COLORS)

from typing import Dict, Optional, Type, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Import PLOTS_DIR for correct plot saving location
from eurus.config import PLOTS_DIR

# Pre-import common scientific libraries for convenience (parent-side only)
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta



# =============================================================================
# PERSISTENT SUBPROCESS REPL
# =============================================================================

# The Python script that runs inside the subprocess.
# It receives JSON commands on stdin and sends JSON responses on stdout.
_SUBPROCESS_SCRIPT = r'''
import sys
import os
import json
import gc
from io import StringIO

# Apply Eurus matplotlib style INSIDE the subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

_style = json.loads(os.environ.get("EURUS_MPL_STYLE", "{}"))
if _style:
    matplotlib.rcParams.update(_style)
_colors = json.loads(os.environ.get("EURUS_MPL_COLORS", "[]"))
if _colors:
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=_colors)

# Pre-import scientific stack
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# Set up execution globals with pre-loaded libraries
exec_globals = {
    "__builtins__": __builtins__,
    "pd": pd,
    "np": np,
    "xr": xr,
    "plt": plt,
    "mcolors": mcolors,
    "cm": cm,
    "datetime": datetime,
    "timedelta": timedelta,
    "PLOTS_DIR": os.environ.get("EURUS_PLOTS_DIR", "plots"),
}

# Signal readiness
print("SUBPROCESS_READY", flush=True)

while True:
    try:
        line = input()
        if line == "EXIT_SUBPROCESS":
            break

        cmd = json.loads(line)

        if cmd["type"] == "exec":
            code = cmd["code"]

            stdout_capture = StringIO()
            stderr_capture = StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Try eval first (expression mode), fall back to exec
                try:
                    compiled = compile(code, "<repl>", "eval")
                    result = eval(compiled, exec_globals)
                    output = stdout_capture.getvalue()
                    if result is not None:
                        output += repr(result)
                    if not output.strip():
                        output = repr(result) if result is not None else "(No output)"
                except SyntaxError:
                    # Jupyter-style: auto-print last expression in multi-line code
                    import ast as _ast
                    try:
                        tree = _ast.parse(code)
                        if tree.body and isinstance(tree.body[-1], _ast.Expr):
                            # Separate the last expression from preceding stmts
                            last_expr_node = tree.body.pop()
                            if tree.body:
                                exec(compile(_ast.Module(body=tree.body, type_ignores=[]), "<repl>", "exec"), exec_globals)
                            result = eval(compile(_ast.Expression(body=last_expr_node.value), "<repl>", "eval"), exec_globals)
                            output = stdout_capture.getvalue()
                            if result is not None:
                                output += repr(result) if not output.strip() else "\n" + repr(result)
                        else:
                            exec(code, exec_globals)
                            output = stdout_capture.getvalue()
                    except SyntaxError:
                        exec(code, exec_globals)
                        output = stdout_capture.getvalue()
                    if not output.strip():
                        output = "(Executed successfully. Use print() to see results.)"

                sys.stdout, sys.stderr = old_stdout, old_stderr
                result_json = {
                    "status": "success",
                    "stdout": output.strip(),
                    "stderr": stderr_capture.getvalue(),
                }

            except Exception as e:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                import traceback
                result_json = {
                    "status": "error",
                    "error": f"Error: {str(e)}\n{traceback.format_exc()}",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                }
            finally:
                plt.close("all")
                gc.collect()

            print(json.dumps(result_json), flush=True)

    except EOFError:
        break
    except Exception as e:
        # Fatal error in the communication loop itself
        old_stdout = sys.__stdout__
        sys.stdout = old_stdout
        print(json.dumps({"status": "fatal", "error": str(e)}), flush=True)
'''


class PersistentREPL:
    """
    Manages a persistent Python subprocess for code execution.
    Provides true process isolation with clean kill on timeout.
    """

    def __init__(self, working_dir: str = "."):
        self._working_dir = working_dir
        self._process: Optional[subprocess.Popen] = None
        self._temp_script: Optional[str] = None
        self._lock = threading.Lock()  # Serialize access per instance
        self._start_subprocess()

    def _start_subprocess(self):
        """Start a new Python subprocess with Eurus environment."""
        # Write the subprocess script to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="eurus_repl_"
        ) as f:
            f.write(_SUBPROCESS_SCRIPT)
            self._temp_script = f.name

        # Build env: inject matplotlib style + PLOTS_DIR
        env = os.environ.copy()
        env["EURUS_MPL_STYLE"] = json.dumps(
            {k: v for k, v in _EURUS_STYLE.items() if isinstance(v, (int, float, str, bool))}
        )
        env["EURUS_MPL_COLORS"] = json.dumps(_EURUS_COLORS)
        env["EURUS_PLOTS_DIR"] = str(PLOTS_DIR)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            [sys.executable, "-u", self._temp_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=self._working_dir if os.path.isdir(self._working_dir) else None,
            env=env,
        )

        # Wait for ready signal
        ready_line = self._process.stdout.readline()
        if "SUBPROCESS_READY" not in ready_line:
            raise RuntimeError(f"Subprocess failed to start: {ready_line!r}")

        logger.info("Started REPL subprocess (PID: %d)", self._process.pid)

    def _ensure_alive(self):
        """Restart subprocess if it has died."""
        if self._process is None or self._process.poll() is not None:
            logger.warning("REPL subprocess died — restarting")
            self._cleanup_process()
            self._start_subprocess()

    def run(self, code: str, timeout: int = 300) -> str:
        """Execute code in the subprocess. Returns output string."""
        with self._lock:
            self._ensure_alive()

            cmd = json.dumps({"type": "exec", "code": code}) + "\n"
            try:
                self._process.stdin.write(cmd)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                logger.error("Subprocess stdin broken: %s — restarting", e)
                self._cleanup_process()
                self._start_subprocess()
                return f"Error: REPL subprocess crashed. Please re-run your code."

            # Read response with timeout
            result_line = self._read_with_timeout(timeout)

            if result_line is None:
                # Timeout — kill subprocess and restart
                logger.warning("REPL execution timed out after %ds — killing subprocess", timeout)
                self._kill_subprocess()
                self._start_subprocess()
                return (
                    "TIMEOUT ERROR: Execution exceeded "
                    f"{timeout} seconds ({timeout // 60} min). "
                    "TIP: Resample data to daily/monthly before plotting "
                    "(e.g., ds.resample(time='D').mean())."
                )

            try:
                result = json.loads(result_line)
            except json.JSONDecodeError:
                return f"Error: Malformed response from subprocess: {result_line!r}"

            if result["status"] == "success":
                output = result.get("stdout", "")
                stderr = result.get("stderr", "")
                if stderr:
                    output = f"{output}\n{stderr}" if output else stderr
                return output or "(No output)"
            elif result["status"] == "error":
                return result.get("error", "Unknown error")
            else:
                return f"Fatal subprocess error: {result.get('error', 'Unknown')}"

    def _read_with_timeout(self, timeout: int) -> Optional[str]:
        """Read one line from subprocess stdout with a timeout."""
        result = [None]

        def _reader():
            try:
                result[0] = self._process.stdout.readline()
            except Exception:
                pass

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()
        reader_thread.join(timeout=timeout)

        if reader_thread.is_alive():
            return None  # Timed out
        return result[0] if result[0] else None

    def _kill_subprocess(self):
        """Force-kill the subprocess."""
        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception as e:
                logger.error("Error killing subprocess: %s", e)
            self._process = None

    def _cleanup_process(self):
        """Clean up subprocess and temp files."""
        self._kill_subprocess()
        if self._temp_script and os.path.exists(self._temp_script):
            try:
                os.unlink(self._temp_script)
            except OSError:
                pass
            self._temp_script = None

    def _update_plots_dir(self, plots_dir: str):
        """Update the PLOTS_DIR used by the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                # Send a command to update the plots directory in the subprocess
                cmd = f"import os; os.environ['EURUS_PLOTS_DIR'] = {plots_dir!r}; PLOTS_DIR = {plots_dir!r}\n"
                self._process.stdin.write(cmd)
                self._process.stdin.flush()
                # Clear the response
                self._read_response(timeout=2)
            except Exception as e:
                logger.warning("Failed to update plots_dir in subprocess: %s", e)

    def close(self):
        """Gracefully shutdown the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write("EXIT_SUBPROCESS\n")
                self._process.stdin.flush()
                self._process.wait(timeout=3)
                logger.info("REPL subprocess exited gracefully (PID: %d)", self._process.pid)
            except Exception:
                self._kill_subprocess()
        self._cleanup_process()


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class PythonREPLInput(BaseModel):
    code: str = Field(description="The Python code to execute.")


class PythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = (
        "A Python REPL for data analysis and visualization.\n\n"
        "CRITICAL PLOTTING RULES:\n"
        "1. ALWAYS save to PLOTS_DIR: plt.savefig(f'{PLOTS_DIR}/filename.png')\n"
        "2. Use descriptive filenames (e.g., 'route_risk_map.png')\n"
        "\n\n"
        "MEMORY RULES:\n"
        "1. NEVER use .load() or .compute() on large datasets\n"
        "2. Resample multi-year data first: ds.resample(time='D').mean()\n"
        "3. Use .sel() to subset data before operations\n\n"
        "Pre-loaded: pd, np, xr, plt, mcolors, cm, datetime, timedelta, PLOTS_DIR (string path)"
    )
    args_schema: Type[BaseModel] = PythonREPLInput
    working_dir: str = "."
    _repl: Optional[PersistentREPL] = None
    _plot_callback: Optional[Callable] = None  # For web interface
    _displayed_plots: set = set()
    _plots_dir: Optional[str] = None  # Session-specific plot directory

    def __init__(self, working_dir: str = ".", plots_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.working_dir = working_dir
        self._plot_callback = None
        self._displayed_plots = set()
        self._plots_dir = plots_dir or str(PLOTS_DIR)
        # Ensure the plots directory exists
        Path(self._plots_dir).mkdir(parents=True, exist_ok=True)
        self._repl = PersistentREPL(working_dir=working_dir)
        # Override the subprocess PLOTS_DIR env var to use session-specific dir
        if plots_dir:
            self._repl._update_plots_dir(plots_dir)

    def set_plot_callback(self, callback: Callable):
        """Set callback for plot capture (used by web interface)."""
        self._plot_callback = callback

    def close(self):
        """Clean up subprocess resources."""
        if self._repl:
            self._repl.close()
            self._repl = None

    def _display_image_in_terminal(self, filepath: str, base64_data: str):
        """Display image in terminal — iTerm2/VSCode inline, or macOS Preview fallback."""
        # Skip if already displayed this file in this session
        if filepath in self._displayed_plots:
            return
        self._displayed_plots.add(filepath)

        try:
            term_program = os.environ.get("TERM_PROGRAM", "")

            # iTerm2 inline image protocol (only iTerm2 supports this)
            if "iTerm.app" in term_program:
                sys.stdout.write(f"\033]1337;File=inline=1;width=auto;preserveAspectRatio=1:{base64_data}\a\n")
                sys.stdout.flush()
                return

            # Fallback: open in Preview on macOS (only in CLI, not web)
            if not self._plot_callback and os.path.exists(filepath):
                import subprocess as _sp
                _sp.Popen(["open", filepath], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

        except Exception as e:
            logger.warning(f"Failed to display image in terminal: {e}")

    def _capture_and_notify_plots(self, saved_files: list, code: str = ""):
        """Capture plots and notify via callback."""
        for filepath in saved_files:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        img_data = f.read()
                    b64_data = base64.b64encode(img_data).decode('utf-8')

                    # Display in terminal
                    self._display_image_in_terminal(filepath, b64_data)

                    # Send to web UI via callback
                    if self._plot_callback:
                        self._plot_callback(b64_data, filepath, code)
            except Exception as e:
                print(f"Warning: Failed to capture plot {filepath}: {e}")

    def _run(self, code: str) -> str:
        """Execute the python code in the subprocess and return the output."""
        plots_dir = self._plots_dir or str(PLOTS_DIR)

        # Snapshot plots directory BEFORE execution
        image_exts = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.gif', '.webp'}
        try:
            before_files = {
                f: os.path.getmtime(os.path.join(plots_dir, f))
                for f in os.listdir(plots_dir)
                if os.path.splitext(f)[1].lower() in image_exts
            }
        except FileNotFoundError:
            before_files = {}

        # Execute in subprocess
        output = self._repl.run(code, timeout=300)

        # Detect NEW plot files by comparing directory snapshots
        try:
            after_files = {
                f: os.path.getmtime(os.path.join(plots_dir, f))
                for f in os.listdir(plots_dir)
                if os.path.splitext(f)[1].lower() in image_exts
            }
        except FileNotFoundError:
            after_files = {}

        new_files = []
        for fname, mtime in after_files.items():
            full_path = os.path.join(plots_dir, fname)
            if fname not in before_files or mtime > before_files[fname]:
                if full_path not in self._displayed_plots:
                    new_files.append(full_path)

        if new_files:
            print(f"📊 {len(new_files)} plot(s) saved")
            self._capture_and_notify_plots(new_files, code)

        return output

    async def _arun(self, code: str) -> str:
        """Use the tool asynchronously — avoids blocking the event loop."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, code)
