import os
from pathlib import Path

# Silence LiteLLM remote cost map fetching and telemetry across all YAW processes and threads
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")

# Default Deno cache to repo-local cache when present
workspace_deno = Path(__file__).resolve().parent.parent / ".yaw" / "deno_dir"
if workspace_deno.exists():
    os.environ.setdefault("DENO_DIR", str(workspace_deno))
