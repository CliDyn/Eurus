"""
Superb Python REPL Tool
=======================
A persistent Python execution environment for the agent.
Supports state preservation, plotting, and data analysis.

PLOT CAPTURE: When running in web mode, plots are captured via callback.
"""

import sys
import io
import gc
import os
import base64
import contextlib
import traceback
import threading  # For global REPL lock
import matplotlib
# Force non-interactive backend to prevent crashes on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional, Type, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Import PLOTS_DIR for correct plot saving location
from vostok.config import PLOTS_DIR

# Pre-import common scientific libraries for convenience
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# Security: Block dangerous imports and builtins
BLOCKED_IMPORTS = ['subprocess', 'socket', 'multiprocessing', 'ctypes']
BLOCKED_PATTERNS = [
    'import os',
    'from os',
    'import sys',
    'from sys',
    'import subprocess',
    'import socket',
    'open(',
    '__import__',
    'exec(',
    'eval(',
]

def _check_security(code: str) -> str | None:
    """Check code for security violations. Returns error message or None."""
    # Check blocked patterns first
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"Security Error: '{pattern.split('(')[0]}' is blocked for safety."
    # Also check BLOCKED_IMPORTS (catches 'from subprocess import X')
    for blocked in BLOCKED_IMPORTS:
        if blocked in code:
            return f"Security Error: '{blocked}' module is blocked for safety."
    return None


# Global lock for matplotlib thread safety
_repl_lock = threading.Lock()


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
        "Pre-loaded: pd, np, xr, plt, datetime, timedelta, PLOTS_DIR (string path)"
    )
    args_schema: Type[BaseModel] = PythonREPLInput
    globals_dict: Dict = Field(default_factory=dict, exclude=True)
    working_dir: str = "."
    _plot_callback: Optional[Callable] = None  # For web interface

    def __init__(self, working_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.working_dir = working_dir
        self._plot_callback = None
        # Initialize globals with SAFE libraries only
        # SECURITY: os/shutil/Path removed - they allow reading arbitrary files
        self.globals_dict = {
            "pd": pd,
            "np": np,
            "xr": xr,
            "plt": plt,
            "datetime": datetime,
            "timedelta": timedelta,
            "PLOTS_DIR": str(PLOTS_DIR),  # STRING only! Path object allows .parent exploit
        }

    def set_plot_callback(self, callback: Callable):
        """Set callback for plot capture (used by web interface)."""
        self._plot_callback = callback
        
    def close(self):
        """Clean up resources."""
        pass  # No kernel to close in simple implementation

    def _capture_and_notify_plots(self, saved_files: list):
        """Capture plots and notify via callback."""
        if not self._plot_callback:
            return
            
        for filepath in saved_files:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        img_data = f.read()
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    self._plot_callback(b64_data, filepath, "")
            except Exception as e:
                print(f"Warning: Failed to capture plot {filepath}: {e}")

    def _run(self, code: str) -> str:
        """Execute the python code and return the output."""
        import threading

        # Security check FIRST
        security_error = _check_security(code)
        if security_error:
            return security_error

        # Use global lock for matplotlib thread safety
        with _repl_lock:
            # Execute with timeout using threading
            result_container = {"output": None, "error": None, "saved_files": []}
            
            # Create wrapper to track saved files
            original_savefig = plt.savefig
            def tracking_savefig(*args, **kwargs):
                result = original_savefig(*args, **kwargs)
                filepath = None
                if args:
                    filepath = str(args[0])
                elif 'fname' in kwargs:  # LLM often uses plt.savefig(fname="x.png")
                    filepath = str(kwargs['fname'])
                
                if filepath:
                    result_container["saved_files"].append(filepath)
                    print(f"📊 Plot saved: {filepath}")
                return result
            
            def execute_code():
                # Thread-safe stdout capture using contextlib
                redirected_output = io.StringIO()
                
                # Inject tracking savefig
                self.globals_dict["plt"].savefig = tracking_savefig
                
                try:
                    # Use redirect_stdout for thread-safe output capture
                    with contextlib.redirect_stdout(redirected_output):
                        # Try to compile as an expression first (like a real REPL)
                        try:
                            compiled = compile(code, '<repl>', 'eval')
                            result = eval(compiled, self.globals_dict)
                            output = redirected_output.getvalue()
                            if result is not None:
                                output += repr(result)
                            result_container["output"] = output.strip() if output.strip() else repr(result) if result is not None else "(No output)"
                        except SyntaxError:
                            # Not an expression, execute as statements
                            exec(code, self.globals_dict)
                            output = redirected_output.getvalue()
                            
                            if not output.strip():
                                result_container["output"] = "(Executed successfully. Use print() to see results.)"
                            else:
                                result_container["output"] = output.strip()
                        
                except Exception as e:
                    result_container["error"] = f"Error: {str(e)}\n{traceback.format_exc()}"
                    
                finally:
                    # Restore original savefig
                    self.globals_dict["plt"].savefig = original_savefig
                    # Close figures AFTER capturing
                    plt.close('all')
                    gc.collect()
            
            # Run in thread with 300-second timeout (5 min) for large data operations
            exec_thread = threading.Thread(target=execute_code)
            exec_thread.start()
            exec_thread.join(timeout=300)

            if exec_thread.is_alive():
                # Thread is still running after timeout
                return "TIMEOUT ERROR: Execution exceeded 300 seconds (5 min). TIP: Resample data to daily/monthly before plotting (e.g., ds.resample(time='D').mean())."
            
            # Capture plots and send to web interface
            if result_container["saved_files"]:
                self._capture_and_notify_plots(result_container["saved_files"])
            
            if result_container["error"]:
                return result_container["error"]
            
            return result_container["output"] or "(No output)"

    async def _arun(self, code: str) -> str:
        """Use the tool asynchronously."""
        return self._run(code)
