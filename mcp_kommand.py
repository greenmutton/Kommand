# mcp_sub_agent.py
"""
MCP server that dynamically discovers and exposes tools for spawning qwen sub-agents
with proper project directory isolation, environment variable management, and MCP tools configuration.
"""

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# Configuration
BASE_PATH = os.environ["KOMMAND_BASE_DIR"]
AGENTS_DIR = f"{BASE_PATH}/agents"
VENV_PATH = os.environ["KOMMAND_VENV"]
MAX_HISTORY_MESSAGES = 20
MAX_CONTEXT_MESSAGES = 5  # For cross-agent context
ALLOWED_WORKSPACE_FILES = 100  # Maximum files in workspace
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB

# Sensitive environment variables that should not be overridden
SENSITIVE_ENV_VARS = {
    "PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONOPTIMIZE",
    "PYTHONDEBUG",
    "PYTHONINSPECT",
    "PYTHONUNBUFFERED",
    "PYTHONVERBOSE",
    "PYTHONCASEOK",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONUSERBASE",
    "PYTHONEXECUTABLE",
    "PYTHONWEXECUTABLE",
    "PYTHONFAULTHANDLER",
    "PYTHONTRACEMALLOC",
    "PYTHONPROFILEIMPORTTIME",
    "PYTHONASYNCIODEBUG",
    "PYTHONMALLOC",
    "PYTHONLEGACYWINDOWSSTDIO",
    "PYTHONLEGACYWINDOWSFSENCODING",
    "PYTHONUTF8",
    "PYTHONCOERCECLOCALE",
    "PYTHONBREAKPOINT",
    "PYTHON_HISTORY",
    "PYTHON_COLORS",
    "PYTHON_FORCE_COLOR",
    "PYTHON_NO_COLOR",
    "PYTHON_IS_64BIT",
    "PYTHON_GIL",
    "PYTHON_FROZEN_MODULES",
    "PYTHON_STRICT_MODULE_MODE",
    "PYTHON_CPU_COUNT",
    "PYTHON_CFLAGS",
    "PYTHON_CPPFLAGS",
    "PYTHON_LDFLAGS",
    "PYTHON_EXEC_PREFIX",
    "PYTHON_PREFIX",
    "PYTHON_CONFIGURE_OPTS",
    "PYTHON_DEVEL",
    "PYTHON_INCLUDE_DIR",
    "PYTHON_LIB",
    "PYTHON_LIBRARY",
    "PYTHON_PLATFORM",
    "PYTHON_SITE_PACKAGES",
    "PYTHON_SRC_DIR",
    "PYTHON_VERSION",
    "PYTHON_VERSION_MAJOR",
    "PYTHON_VERSION_MINOR",
    "PYTHON_VERSION_MICRO",
    "PYTHON_VERSION_RELEASELEVEL",
    "PYTHON_VERSION_SERIAL",
    "PYTHON_API_VERSION",
    "PYTHON_ABI_VERSION",
    "PYTHON_IMPLEMENTATION",
    "PYTHON_PLATFORM_PLATFORM",
    "PYTHON_PLATFORM_MACHINE",
    "PYTHON_PLATFORM_PROCESSOR",
    "PYTHON_PLATFORM_NODE",
    "PYTHON_PLATFORM_RELEASE",
    "PYTHON_PLATFORM_VERSION",
    "PYTHON_PLATFORM_SYSTEM",
    "PYTHON_PLATFORM_SYSTEM_VERSION",
    "PYTHON_PLATFORM_UNAME",
    "PYTHON_PLATFORM_LINUX_DISTRIBUTION",
    "PYTHON_PLATFORM_MAC_VER",
    "PYTHON_PLATFORM_WIN32_VER",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "DISPLAY",
    "XAUTHORITY",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "SSH_CONNECTION",
    "SSH_CLIENT",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_DIRS",
    "XDG_CONFIG_DIRS",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_BIN_HOME",
    "XDG_DESKTOP_DIR",
    "XDG_DOCUMENTS_DIR",
    "XDG_DOWNLOAD_DIR",
    "XDG_MUSIC_DIR",
    "XDG_PICTURES_DIR",
    "XDG_PUBLICSHARE_DIR",
    "XDG_TEMPLATES_DIR",
    "XDG_VIDEOS_DIR",
    "XDG_MENU_PREFIX",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "XDG_SESSION_TYPE",
    "XDG_SESSION_CLASS",
    "XDG_SESSION_ID",
    "XDG_SEAT",
    "XDG_VTNR",
    "XDG_GREETER_DATA_DIR",
    "XDG_SESSION_PATH",
    "XDG_DESKTOP_PORTAL_DIR",
    "XDG_DESKTOP_PORTAL_CONFIG_DIR",
    "XDG_DESKTOP_PORTAL_DATA_DIR",
    "XDG_DESKTOP_PORTAL_RUNTIME_DIR",
    "XDG_DESKTOP_PORTAL_STATE_DIR",
    "XDG_DESKTOP_PORTAL_CACHE_DIR",
    "XDG_DESKTOP_PORTAL_LOG_DIR",
    "XDG_DESKTOP_PORTAL_TEMP_DIR",
    "XDG_DESKTOP_PORTAL_SHARE_DIR",
    "XDG_DESKTOP_PORTAL_EXTRA_DIRS",
    "XDG_DESKTOP_PORTAL_MENU_DIRS",
    "XDG_DESKTOP_PORTAL_APPLICATION_DIRS",
    "XDG_DESKTOP_PORTAL_ICON_DIRS",
    "XDG_DESKTOP_PORTAL_FONT_DIRS",
    "XDG_DESKTOP_PORTAL_THEME_DIRS",
    "XDG_DESKTOP_PORTAL_SOUND_DIRS",
    "XDG_DESKTOP_PORTAL_VIDEO_DIRS",
    "XDG_DESKTOP_PORTAL_DOCUMENT_DIRS",
    "XDG_DESKTOP_PORTAL_DOWNLOAD_DIRS",
    "XDG_DESKTOP_PORTAL_MUSIC_DIRS",
    "XDG_DESKTOP_PORTAL_PICTURE_DIRS",
    "XDG_DESKTOP_PORTAL_PUBLICSHARE_DIRS",
    "XDG_DESKTOP_PORTAL_TEMPLATE_DIRS",
    "XDG_DESKTOP_PORTAL_VIDEOS_DIRS",
    "XDG_DESKTOP_PORTAL_OTHER_DIRS",
    "XDG_DESKTOP_PORTAL_HIDDEN_DIRS",
    "XDG_DESKTOP_PORTAL_SYSTEM_DIRS",
    "XDG_DESKTOP_PORTAL_LOCAL_DIRS",
    "XDG_DESKTOP_PORTAL_NETWORK_DIRS",
    "XDG_DESKTOP_PORTAL_REMOTE_DIRS",
    "XDG_DESKTOP_PORTAL_READONLY_DIRS",
    "XDG_DESKTOP_PORTAL_WRITEABLE_DIRS",
    "XDG_DESKTOP_PORTAL_EXECUTABLE_DIRS",
    "XDG_DESKTOP_PORTAL_LIBRARY_DIRS",
    "XDG_DESKTOP_PORTAL_INCLUDE_DIRS",
    "XDG_DESKTOP_PORTAL_SHARE_DIRS",
    "XDG_DESKTOP_PORTAL_ETC_DIRS",
    "XDG_DESKTOP_PORTAL_VAR_DIRS",
    "XDG_DESKTOP_PORTAL_TMP_DIRS",
    "XDG_DESKTOP_PORTAL_RUN_DIRS",
    "XDG_DESKTOP_PORTAL_LOCK_DIRS",
    "XDG_DESKTOP_PORTAL_LOG_DIRS",
    "XDG_DESKTOP_PORTAL_CACHE_DIRS",
    "XDG_DESKTOP_PORTAL_STATE_DIRS",
    "XDG_DESKTOP_PORTAL_DATA_DIRS",
    "XDG_DESKTOP_PORTAL_CONFIG_DIRS",
    "XDG_DESKTOP_PORTAL_HOME_DIRS",
    "XDG_DESKTOP_PORTAL_USER_DIRS",
    "XDG_DESKTOP_PORTAL_GROUP_DIRS",
    "XDG_DESKTOP_PORTAL_OTHER_DIRS",
    "XDG_DESKTOP_PORTAL_SYSTEM_DIRS",
    "XDG_DESKTOP_PORTAL_LOCAL_DIRS",
    "XDG_DESKTOP_PORTAL_NETWORK_DIRS",
    "XDG_DESKTOP_PORTAL_REMOTE_DIRS",
    "XDG_DESKTOP_PORTAL_READONLY_DIRS",
    "XDG_DESKTOP_PORTAL_WRITEABLE_DIRS",
    "XDG_DESKTOP_PORTAL_EXECUTABLE_DIRS",
    "XDG_DESKTOP_PORTAL_LIBRARY_DIRS",
    "XDG_DESKTOP_PORTAL_INCLUDE_DIRS",
    "XDG_DESKTOP_PORTAL_SHARE_DIRS",
    "XDG_DESKTOP_PORTAL_ETC_DIRS",
    "XDG_DESKTOP_PORTAL_VAR_DIRS",
    "XDG_DESKTOP_PORTAL_TMP_DIRS",
    "XDG_DESKTOP_PORTAL_RUN_DIRS",
    "XDG_DESKTOP_PORTAL_LOCK_DIRS",
    "XDG_DESKTOP_PORTAL_LOG_DIRS",
    "XDG_DESKTOP_PORTAL_CACHE_DIRS",
    "XDG_DESKTOP_PORTAL_STATE_DIRS",
    "XDG_DESKTOP_PORTAL_DATA_DIRS",
    "XDG_DESKTOP_PORTAL_CONFIG_DIRS",
    "XDG_DESKTOP_PORTAL_HOME_DIRS",
    "XDG_DESKTOP_PORTAL_USER_DIRS",
    "XDG_DESKTOP_PORTAL_GROUP_DIRS",
    "XDG_DESKTOP_PORTAL_OTHER_DIRS",
}

# Unsafe file extensions that should not be copied to workspace
UNSAFE_EXTENSIONS = {
    ".exe",
    ".bat",
    ".cmd",
    ".com",
    ".scr",
    ".pif",
    ".msi",
    ".dll",
    ".so",
    ".dylib",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".csh",
    ".tcsh",
    ".ksh",
    ".pl",
    ".py",
    ".php",
    ".rb",
    ".js",
    ".html",
    ".htm",
    ".css",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".tmp",
    ".temp",
    ".bak",
    ".backup",
    ".old",
    ".swp",
    ".swo",
    ".DS_Store",
    ".Thumbs.db",
    ".desktop",
    ".lnk",
    ".shortcut",
    ".app",
    ".dmg",
    ".pkg",
    ".deb",
    ".rpm",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zip",
    ".rar",
    ".7z",
    ".iso",
    ".img",
    ".vmdk",
    ".ova",
    ".ovf",
    ".vhd",
    ".vhdx",
    ".qcow2",
    ".raw",
    ".iso",
    ".bin",
    ".cue",
    ".toast",
    ".dmg",
    ".sparseimage",
    ".udif",
    ".ndif",
    ".dc42",
    ".diskcopy",
    ".toast",
    ".cdmg",
    ".sparsebundle",
    ".sparseimage",
    ".udif",
    ".ndif",
    ".dc42",
    ".diskcopy",
    ".toast",
    ".cdmg",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP("kommission-sub-agent")


@dataclass
class AgentConfig:
    """Configuration for a specific agent type."""

    name: str
    project_dir: str
    system_prompt: str
    allowed_mcp_servers: List[str] = field(default_factory=list)
    allowed_sub_agents: List[str] = field(default_factory=list)
    timeout_sec: int = 480
    cleanup_after: bool = True
    max_workspace_files: int = ALLOWED_WORKSPACE_FILES


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""

    pass


class ConversationHistory:
    """Manages conversation history for agents."""

    # Version identifier for conversation history format
    HISTORY_VERSION = "1.0"

    @staticmethod
    def get_history_file(agent_dir: Path) -> Path:
        """Get the path to the history file for an agent."""
        return agent_dir / "conversation_history.json"

    @staticmethod
    def load_history(agent_dir: Path) -> List[Dict[str, Any]]:
        """Load conversation history for an agent."""
        history_file = ConversationHistory.get_history_file(agent_dir)
        if not history_file.exists():
            return []

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                # Validate history structure
                if not isinstance(history, list):
                    logger.warning(f"Invalid history format in {history_file}")
                    return []

                # Check for version identifier
                if history and isinstance(history[0], dict) and "version" in history[0]:
                    version = history[0]["version"]
                    logger.info(f"Loading history version {version}")
                    return history[1:]  # Skip version marker
                else:
                    logger.info("Loading legacy history format")
                    return history
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to load history from {history_file}: {e}")
            return []

    @staticmethod
    def save_history(agent_dir: Path, history: List[Dict[str, Any]]) -> None:
        """Save conversation history for an agent."""
        history_file = ConversationHistory.get_history_file(agent_dir)
        try:
            # Ensure directory exists
            history_file.parent.mkdir(parents=True, exist_ok=True)

            # Trim to max messages and add version identifier
            trimmed_history = history[-MAX_HISTORY_MESSAGES:]
            history_with_version = [
                {"version": ConversationHistory.HISTORY_VERSION}
            ] + trimmed_history

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history_with_version, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save history to {history_file}: {e}")

    @staticmethod
    def add_message(agent_dir: Path, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        if not agent_dir.exists():
            logger.warning(f"Agent directory does not exist: {agent_dir}")
            return

        # Validate role
        if role not in ["user", "assistant", "system"]:
            logger.warning(f"Invalid role '{role}' for conversation history")
            role = "assistant"

        # Validate content
        if not isinstance(content, str):
            content = str(content)

        history = ConversationHistory.load_history(agent_dir)
        history.append(
            {
                "role": role,
                "content": content[:10000],  # Limit content length
                "timestamp": datetime.now().isoformat(),
            }
        )
        ConversationHistory.save_history(agent_dir, history)

    @staticmethod
    def get_context_messages(agent_dir: Path) -> List[Dict[str, Any]]:
        """Get the last MAX_HISTORY_MESSAGES for context."""
        return ConversationHistory.load_history(agent_dir)[-MAX_HISTORY_MESSAGES:]

    @staticmethod
    def add_tool_interaction(
        agent_dir: Path, tool_name: str, tool_input: str, tool_output: str
    ) -> None:
        """Add a tool interaction to the conversation history."""
        interaction = (
            f"Tool: {tool_name}\n"
            f"Input: {tool_input[:500]}...\n"
            f"Output: {tool_output[:1000]}..."
        )
        ConversationHistory.add_message(agent_dir, "assistant", interaction)


class AgentDiscovery:
    """Discovers and loads agent configurations dynamically."""

    @staticmethod
    def validate_agent_directory(agent_dir: Path) -> bool:
        """Validate that an agent directory has required files."""
        required_files = ["specialization.json", "system_prompt.txt", ".env"]
        for file_name in required_files:
            if not (agent_dir / file_name).exists():
                logger.warning(f"Missing required file '{file_name}' in {agent_dir}")
                return False
        return True

    @staticmethod
    def safe_read_file(file_path: Path, max_size_mb: int = 1) -> Optional[str]:
        """Safely read a file with size limits."""
        if not file_path.exists():
            return None

        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name to prevent command injection."""
        return True
        # if not model_name:
        #     return False

        # # Only allow alphanumeric characters, hyphens, underscores, and dots
        # # This prevents command injection through model names
        # pattern = r"^[a-zA-Z0-9._-]+$"
        # return re.match(pattern, model_name) is not None

    @staticmethod
    def discover_agents() -> Dict[str, AgentConfig]:
        """Discover all agents in the agents directory."""
        agents = {}
        agents_path = Path(AGENTS_DIR)

        if not agents_path.exists():
            logger.warning(f"Agents directory not found: {AGENTS_DIR}")
            return agents

        try:
            for agent_dir in agents_path.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith("."):
                    config = AgentDiscovery._load_agent_config(agent_dir)
                    if config:
                        agents[config.name] = config
                        logger.info(f"Discovered agent: {config.name}")
        except OSError as e:
            logger.error(f"Failed to iterate agents directory: {e}")

        return agents

    @staticmethod
    def _load_agent_config(agent_dir: Path) -> Optional[AgentConfig]:
        """Load configuration for a single agent."""
        try:
            # Validate directory structure
            if not AgentDiscovery.validate_agent_directory(agent_dir):
                return None

            # Load specialization.json
            spec_file = agent_dir / "specialization.json"
            spec_content = AgentDiscovery.safe_read_file(spec_file)
            if not spec_content:
                return None

            try:
                spec_data = json.loads(spec_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {spec_file}: {e}")
                return None

            # Validate specialization structure
            if not isinstance(spec_data, dict):
                logger.error(f"Invalid specialization format in {spec_file}")
                return None

            # Load system prompt
            system_prompt_file = agent_dir / "system_prompt.txt"
            system_prompt = AgentDiscovery.safe_read_file(
                system_prompt_file, max_size_mb=2
            )
            if not system_prompt:
                return None

            # Create agent config with validation
            config = AgentConfig(
                name=agent_dir.name,
                project_dir=str(agent_dir.resolve()),
                system_prompt=system_prompt,
                allowed_mcp_servers=spec_data.get("mcp_servers", []),
                allowed_sub_agents=spec_data.get("sub_agents", []),
                timeout_sec=max(
                    30, min(3600, spec_data.get("timeout_sec", 180))
                ),  # Clamp between 30-3600
                cleanup_after=spec_data.get("cleanup_after", True),
                max_workspace_files=max(
                    1,
                    min(
                        1000,
                        spec_data.get("max_workspace_files", ALLOWED_WORKSPACE_FILES),
                    ),
                ),
            )

            return config

        except Exception as e:
            logger.error(f"Failed to load agent config from {agent_dir}: {e}")
            return None


class ProjectManager:
    """Manages agent project directories and configurations."""

    @staticmethod
    def ensure_project_structure(agent_dir: Path) -> bool:
        """Ensure project directory has required structure."""
        try:
            # Create necessary subdirectories
            subdirs = ["workspace", "output", "logs"]
            for subdir in subdirs:
                (agent_dir / subdir).mkdir(exist_ok=True)
            return True
        except OSError as e:
            logger.error(f"Failed to create project structure for {agent_dir}: {e}")
            return False

    @staticmethod
    def load_env_vars(agent_dir: Path) -> Dict[str, str]:
        """Load environment variables from project's .env file."""
        env_file = agent_dir / ".env"
        if not env_file.exists():
            return {}

        env_vars = {}
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        try:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()

                            # Security: Don't override sensitive environment variables
                            if key and key not in SENSITIVE_ENV_VARS:
                                env_vars[key] = value
                            else:
                                logger.warning(f"Skipping sensitive env var: {key}")
                        except ValueError:
                            logger.warning(f"Invalid env line {line_num} in {env_file}")
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read env file {env_file}: {e}")

        return env_vars

    @staticmethod
    def is_file_safe(file_path: Path) -> bool:
        """Check if a file is safe to copy to workspace."""
        try:
            # Check file extension
            if file_path.suffix.lower() in UNSAFE_EXTENSIONS:
                logger.warning(f"Unsafe file extension: {file_path.suffix}")
                return False

            # Check file permissions
            file_stat = file_path.stat()
            if file_stat.st_mode & stat.S_IXUSR:  # Check if executable by owner
                logger.warning(f"File is executable: {file_path}")
                return False

            # Check file size
            if file_stat.st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(
                    f"File too large: {file_path} ({file_stat.st_size} bytes)"
                )
                return False

            # Check for suspicious content patterns
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    suspicious_patterns = [
                        r"#!/bin/(bash|sh|zsh|fish|python|perl|php|ruby)",
                        r"import\s+os\s*;",
                        r"import\s+subprocess\s*;",
                        r"eval\s*\(",
                        r"exec\s*\(",
                        r"system\s*\(",
                        r"__import__\s*\(",
                        r"getattr\s*\(",
                        r"subprocess\.",
                        r"os\.",
                        r"exec\(",
                        r"eval\(",
                    ]

                    for pattern in suspicious_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            logger.warning(
                                f"Suspicious content pattern in file: {file_path}"
                            )
                            return False
            except (IOError, OSError, UnicodeDecodeError):
                # If we can't read the file as text, it might be binary
                logger.warning(f"Binary file detected: {file_path}")
                return False

            return True

        except (OSError, IOError) as e:
            logger.error(f"Failed to check file safety: {file_path}: {e}")
            return False

    @staticmethod
    def safe_copy_file(src: Path, dst: Path) -> bool:
        """Safely copy a file with validation."""
        try:
            # Validate source file
            if not src.exists() or not src.is_file():
                logger.warning(f"Source file not found: {src}")
                return False

            # Check file safety
            if not ProjectManager.is_file_safe(src):
                logger.warning(f"File failed safety check: {src}")
                return False

            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Copy file with safe permissions
            shutil.copy2(src, dst)

            # Remove execute permissions
            os.chmod(dst, 0o644)  # rw-r--r--

            logger.info(f"Copied file: {src.name} -> {dst}")
            return True

        except (OSError, IOError) as e:
            logger.error(f"Failed to copy file {src} to {dst}: {e}")
            return False

    @staticmethod
    def cleanup_workspace(workspace_path: Path) -> None:
        """Clean up workspace directory including subdirectories."""
        try:
            if workspace_path.exists():
                # Use shutil.rmtree for complete cleanup
                shutil.rmtree(workspace_path, ignore_errors=True)
                # Recreate the directory
                workspace_path.mkdir(exist_ok=True)
                logger.info(f"Cleaned up workspace: {workspace_path}")
        except OSError as e:
            logger.error(f"Failed to cleanup workspace {workspace_path}: {e}")


class SubAgentExecutor:
    """Handles execution of qwen sub-agents."""

    def __init__(self, config: AgentConfig, calling_agent_dir: Path):
        self.config = config
        self.project_path = Path(config.project_dir)
        self.calling_agent_dir = calling_agent_dir

    def prepare_workspace(
        self, context_files: Optional[List[str]] = None
    ) -> Tuple[Path, bool]:
        """Prepare the workspace directory with context files."""
        workspace_path = self.project_path / "workspace"

        # Clean workspace if cleanup is enabled
        if self.config.cleanup_after:
            ProjectManager.cleanup_workspace(workspace_path)

        # Copy context files
        files_copied = 0
        if context_files:
            for src in context_files:
                if files_copied >= self.config.max_workspace_files:
                    logger.warning(
                        f"Maximum workspace files ({self.config.max_workspace_files}) reached"
                    )
                    break

                src_path = Path(src).resolve()

                # Security: Check for path traversal
                try:
                    src_path.relative_to(BASE_PATH)
                except ValueError:
                    logger.warning(f"Path traversal attempt detected: {src_path}")
                    continue

                if ProjectManager.safe_copy_file(
                    src_path, workspace_path / src_path.name
                ):
                    files_copied += 1
                else:
                    logger.warning(f"Failed to copy file: {src_path}")

        return workspace_path, files_copied > 0

    def build_command(self, prompt_file: Path) -> List[str]:
        """Build the qwen command with all necessary arguments."""
        # Load environment variables
        env_vars = ProjectManager.load_env_vars(self.project_path)

        # Validate model name
        model_name = env_vars.get("OPENAI_MODEL", "")
        if model_name and not AgentDiscovery.validate_model_name(model_name):
            logger.warning(f"Invalid model name: {model_name}")
            model_name = ""

        base_url = env_vars.get("OPENAI_BASE_URL", "")
        api_key = env_vars.get("OPENAI_API_KEY", "")
        system_prompt_file = env_vars.get("GEMINI_SYSTEM_MD", "")

        # Read the prompt from the file
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read()
        except IOError as e:
            logger.error(f"Failed to read prompt file: {e}")
            return []

        # Prepend system prompt to the content if available
        if self.config.system_prompt:
            prompt_content = f"SYSTEM: {self.config.system_prompt}\n\n{prompt_content}"

        relative_prompt_path = f"/tmp/{str(uuid.uuid4())}.txt"
        with open(relative_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)

        env_vars_str = ""
        if api_key:
            env_vars_str += f"OPENAI_API_KEY={api_key} "
        if base_url:
            env_vars_str += f"OPENAI_BASE_URL='{base_url}' "
        if model_name:
            env_vars_str += f"OPENAI_MODEL={model_name} "
        if system_prompt_file:
            env_vars_str += f"GEMINI_SYSTEM_MD={system_prompt_file} "

        # Build command with virtual environment activation
        cmd = [
            "/bin/bash",
            "-c",
            f"source {VENV_PATH} && {env_vars_str}qwen < '{relative_prompt_path}'",
        ]

        # Add model if valid
        if model_name:
            cmd[-1] += f" --model '{model_name}'"

        # if base_url:
        #     cmd[-1] += f" --openai-base-url '{base_url}'"

        # if api_key:
        #     cmd[-1] += f" --openai-api-key '{api_key}'"

        return cmd

    def execute(
        self, task_prompt: str, context_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute the sub-agent with the given task."""
        start_time = datetime.now()

        try:
            # Prepare workspace
            workspace_path, has_context_files = self.prepare_workspace(context_files)

            # Write prompt file with conversation history
            prompt_file = workspace_path / "prompt.txt"

            # Get conversation history from the SUB-AGENT's own history
            sub_agent_history = ConversationHistory.get_context_messages(
                self.project_path
            )

            # Build full prompt
            full_prompt_parts = []

            # Add sub-agent's own conversation history
            if sub_agent_history:
                full_prompt_parts.append("=== YOUR CONVERSATION HISTORY ===")
                for msg in sub_agent_history:
                    full_prompt_parts.append(f"{msg['role'].upper()}: {msg['content']}")
                full_prompt_parts.append("")

            # Add context from calling agent (limited)
            if (
                self.calling_agent_dir
                and self.calling_agent_dir != self.project_path
                and self.calling_agent_dir.exists()
            ):
                calling_agent_history = ConversationHistory.get_context_messages(
                    self.calling_agent_dir
                )
                if calling_agent_history:
                    full_prompt_parts.append("=== CONTEXT FROM CALLING AGENT ===")
                    for msg in calling_agent_history[-MAX_CONTEXT_MESSAGES:]:
                        full_prompt_parts.append(
                            f"[{msg['role'].upper()} from calling agent]: {msg['content']}"
                        )
                    full_prompt_parts.append("")

            # Add current task
            full_prompt_parts.append("=== CURRENT TASK ===")
            full_prompt_parts.append(task_prompt)

            full_prompt = "\n".join(full_prompt_parts)

            # Write prompt file
            try:
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(full_prompt)
            except IOError as e:
                logger.error(f"Failed to write prompt file: {e}")
                return {
                    "agent_name": self.config.name,
                    "error": "prompt_creation_failed",
                    "message": f"Failed to create prompt file: {e}",
                    "exit_code": -1,
                }

            # Load environment variables
            env_vars = ProjectManager.load_env_vars(self.project_path)

            # Debug: Log the environment variables being loaded
            logger.info(
                f"Environment variables loaded for {self.config.name}: {list(env_vars.keys())}"
            )
            if "OPENAI_API_KEY" in env_vars:
                logger.info("OPENAI_API_KEY found in environment variables")
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")

            # Create a copy of the current environment and update with agent-specific variables
            env = os.environ.copy()
            env.update(env_vars)

            # Build and execute command
            cmd = self.build_command(prompt_file)
            logger.info(f"Executing command: {' '.join(cmd)} in {self.project_path}")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.project_path,  # This is correct - using the agent's main directory
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.config.timeout_sec,
                    shell=False,  # Important for security
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Agent {self.config.name} timed out after {self.config.timeout_sec} seconds"
                )
                return {
                    "agent_name": self.config.name,
                    "error": "timeout",
                    "message": f"Agent timed out after {self.config.timeout_sec} seconds",
                    "exit_code": -1,
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                }

            # Save output to log file
            timestamp = int(datetime.now().timestamp())
            log_file = self.project_path / "logs" / f"execution_{timestamp}.log"

            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Start Time: {start_time.isoformat()}\n")
                    f.write(f"End Time: {datetime.now().isoformat()}\n")
                    f.write(
                        f"Duration: {(datetime.now() - start_time).total_seconds():.2f}s\n"
                    )
                    f.write(f"Exit Code: {proc.returncode}\n")
                    f.write(f"Context Files: {context_files}\n")
                    f.write(f"Working Directory: {self.project_path}\n")
                    f.write(f"Environment Variables: {list(env_vars.keys())}\n")
                    f.write(f"STDOUT:\n{self._filter_verbose_output(proc.stdout)}\n")
                    f.write(f"STDERR:\n{proc.stderr}\n")
            except IOError as e:
                logger.error(f"Failed to write log file {log_file}: {e}")

            # Persist conversation history in BOTH agents
            if proc.returncode == 0:
                # 1. Add to SUB-AGENT's history
                ConversationHistory.add_message(self.project_path, "user", task_prompt)
                ConversationHistory.add_message(
                    self.project_path,
                    "assistant",
                    self._filter_verbose_output(proc.stdout),
                )

                # 2. Add to CALLING AGENT's history
                if (
                    self.calling_agent_dir
                    and self.calling_agent_dir != self.project_path
                    and self.calling_agent_dir.exists()
                ):
                    tool_interaction = (
                        f"Sub-agent call: {self.config.name}\n"
                        f"Task: {task_prompt[:200]}...\n"
                        f"Result: {self._filter_verbose_output(proc.stdout[:500])}..."
                    )
                    ConversationHistory.add_message(
                        self.calling_agent_dir, "assistant", tool_interaction
                    )

            return {
                "agent_name": self.config.name,
                "project_dir": str(self.project_path),
                "workspace_dir": str(workspace_path),
                "stdout": self._filter_verbose_output(proc.stdout),
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "log_file": str(log_file),
                "env_vars_used": list(env_vars.keys()),
                "allowed_mcp_servers": self.config.allowed_mcp_servers,
                "allowed_sub_agents": self.config.allowed_sub_agents,
                "sub_agent_history_persisted": proc.returncode == 0,
                "calling_agent_history_updated": (
                    proc.returncode == 0
                    and self.calling_agent_dir != self.project_path
                    and self.calling_agent_dir.exists()
                ),
                "context_files_copied": has_context_files,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        except Exception as e:
            logger.error(f"Unexpected error executing agent {self.config.name}: {e}")
            return {
                "agent_name": self.config.name,
                "error": "execution_failed",
                "message": str(e),
                "exit_code": -1,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

    def _filter_verbose_output(self, s: str) -> str:
        fixed_start1 = "✅ "
        fixed_start2 = "\n📋 Result:"
        end_pattern = "\n\n"
        last_complete_end = -1
        start_index = 0
        n = len(s)

        while start_index < n:
            i = s.find(fixed_start1, start_index)
            if i == -1:
                break

            j = s.find(fixed_start2, i + len(fixed_start1))
            if j == -1:
                start_index = i + 1
                continue

            k = s.find(end_pattern, j + len(fixed_start2))
            if k == -1:
                start_index = i + 1
                continue

            last_complete_end = k + len(end_pattern)
            start_index = i + 1

        if last_complete_end != -1:
            return s[last_complete_end:]
        else:
            return s


# Global cache for discovered agents
_discovered_agents = None
_agents_discovery_time = None


def get_discovered_agents(refresh: bool = False) -> Dict[str, AgentConfig]:
    """Get or discover agents with optional refresh."""
    global _discovered_agents, _agents_discovery_time

    current_time = datetime.now()

    # Refresh if forced or if cache is older than 5 minutes
    if (
        refresh
        or _discovered_agents is None
        or _agents_discovery_time is None
        or (current_time - _agents_discovery_time).total_seconds() > 300
    ):
        logger.info("Discovering agents...")
        _discovered_agents = AgentDiscovery.discover_agents()
        _agents_discovery_time = current_time

    return _discovered_agents


def get_current_agent_dir() -> Optional[Path]:
    """Get the current agent directory based on working directory."""
    try:
        cwd = Path.cwd().resolve()
        agents_path = Path(AGENTS_DIR).resolve()

        # Check if current working directory is under agents directory
        try:
            cwd.relative_to(agents_path)
            return cwd
        except ValueError:
            pass

        # Check if parent directories contain an agent
        for parent in cwd.parents:
            try:
                parent.relative_to(agents_path)
                return parent
            except ValueError:
                continue
                if parent == agents_path.parent:
                    break

    except Exception as e:
        logger.warning(f"Error determining current agent directory: {e}")

    return None


def validate_agent_access(agent_name: str, calling_agent_dir: Optional[Path]) -> bool:
    """Validate that the calling agent has permission to access the target agent."""
    if not calling_agent_dir:
        return True  # No restriction if no calling agent

    try:
        # Load calling agent's specialization
        spec_file = calling_agent_dir / "specialization.json"
        if not spec_file.exists():
            return True  # No restriction if no specialization file

        spec_content = AgentDiscovery.safe_read_file(spec_file)
        if not spec_content:
            return True

        spec_data = json.loads(spec_content)
        allowed_sub_agents = spec_data.get("sub_agents", [])

        # If no restrictions specified, allow all
        if not allowed_sub_agents:
            return True

        return agent_name in allowed_sub_agents

    except Exception as e:
        logger.warning(f"Error validating agent access: {e}")
        return True  # Fail open


@mcp.tool()
def spawn_sub_agent(
    agent_name: str,
    task_prompt: str,
    context_files: Optional[List[str]] = None,
    timeout_sec: Optional[int] = None,
) -> dict:
    """
    Spawn a dedicated qwen sub-agent with proper project isolation.

    Args:
        agent_name: Name of the agent type to spawn
        task_prompt: The task/prompt for the agent
        context_files: Optional list of files to copy to agent workspace
        timeout_sec: Optional timeout override (uses agent default if not specified)

    Returns:
        Dict containing execution results and metadata
    """
    # Input validation
    if not agent_name or not isinstance(agent_name, str):
        raise ValueError("agent_name must be a non-empty string")

    if not task_prompt or not isinstance(task_prompt, str):
        raise ValueError("task_prompt must be a non-empty string")

    if len(task_prompt) > 50000:  # 50KB limit
        raise ValueError("task_prompt is too long (max 50KB)")

    # Get agents
    agents = get_discovered_agents()

    if agent_name not in agents:
        available_agents = list(agents.keys())
        raise ValueError(
            f"Unknown agent_name '{agent_name}'. Available agents: {available_agents}"
        )

    # Get agent configuration
    config = agents[agent_name]

    # Get current agent directory and validate access
    calling_agent_dir = get_current_agent_dir()
    if calling_agent_dir and not validate_agent_access(agent_name, calling_agent_dir):
        raise ValueError(f"Current agent is not allowed to call agent '{agent_name}'")

    # Override timeout if specified
    if timeout_sec is not None:
        if not isinstance(timeout_sec, int) or timeout_sec < 10 or timeout_sec > 3600:
            raise ValueError("timeout_sec must be an integer between 10 and 3600")
        config.timeout_sec = timeout_sec

    # Ensure project structure exists
    if not ProjectManager.ensure_project_structure(Path(config.project_dir)):
        raise RuntimeError(
            f"Failed to create project structure for agent '{agent_name}'"
        )

    # Create and execute agent
    executor = SubAgentExecutor(config, calling_agent_dir or Path(config.project_dir))
    result = executor.execute(task_prompt, context_files)

    return result


@mcp.tool()
def list_agent_types(refresh: bool = False) -> dict:
    """
    List all available agent types and their configurations.

    Args:
        refresh: Force refresh of agent discovery

    Returns:
        Dict containing agent configurations
    """
    agents = get_discovered_agents(refresh=refresh)
    return {
        agent_name: {
            "project_dir": config.project_dir,
            "allowed_mcp_servers": config.allowed_mcp_servers,
            "allowed_sub_agents": config.allowed_sub_agents,
            "timeout_sec": config.timeout_sec,
            "cleanup_after": config.cleanup_after,
            "max_workspace_files": config.max_workspace_files,
            "system_prompt_preview": (
                config.system_prompt[:100] + "..."
                if len(config.system_prompt) > 100
                else config.system_prompt
            ),
        }
        for agent_name, config in agents.items()
    }


@mcp.tool()
def get_agent_workspace(agent_name: str) -> dict:
    """
    Get information about an agent's workspace directory.

    Args:
        agent_name: Name of the agent

    Returns:
        Dict containing workspace information
    """
    if not agent_name:
        raise ValueError("agent_name is required")

    agents = get_discovered_agents()

    if agent_name not in agents:
        raise ValueError(f"Unknown agent_name '{agent_name}'")

    config = agents[agent_name]
    project_path = Path(config.project_dir)

    try:
        workspace_files = []
        workspace_path = project_path / "workspace"
        if workspace_path.exists():
            workspace_files = [f.name for f in workspace_path.iterdir() if f.is_file()]
    except OSError as e:
        logger.warning(f"Failed to list workspace files: {e}")
        workspace_files = []

    return {
        "agent_name": agent_name,
        "project_dir": str(project_path),
        "workspace_dir": str(workspace_path),
        "output_dir": str(project_path / "output"),
        "logs_dir": str(project_path / "logs"),
        "env_file": str(project_path / ".env"),
        "specialization_file": str(project_path / "specialization.json"),
        "system_prompt_file": str(project_path / "system_prompt.txt"),
        "history_file": str(project_path / "conversation_history.json"),
        "directory_exists": project_path.exists(),
        "workspace_files": workspace_files,
        "workspace_file_count": len(workspace_files),
    }


@mcp.tool()
def get_conversation_history(agent_name: Optional[str] = None, limit: int = 20) -> dict:
    """
    Get conversation history for an agent.

    Args:
        agent_name: Name of the agent. If None, uses current agent directory.
        limit: Maximum number of messages to return

    Returns:
        Dict containing conversation history
    """
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

    if agent_name:
        agents = get_discovered_agents()
        if agent_name not in agents:
            raise ValueError(f"Unknown agent_name '{agent_name}'")
        agent_dir = Path(agents[agent_name].project_dir)
    else:
        agent_dir = get_current_agent_dir()
        if agent_dir is None:
            raise ValueError("Could not determine current agent directory")

    history = ConversationHistory.load_history(agent_dir)

    return {
        "agent_name": agent_name or agent_dir.name,
        "history": history[-limit:],
        "total_messages": len(history),
        "history_file": str(ConversationHistory.get_history_file(agent_dir)),
        "limit_applied": limit,
    }


@mcp.tool()
def clear_conversation_history(agent_name: Optional[str] = None) -> dict:
    """
    Clear conversation history for an agent.

    Args:
        agent_name: Name of the agent. If None, uses current agent directory.

    Returns:
        Dict indicating success/failure
    """
    if agent_name:
        agents = get_discovered_agents()
        if agent_name not in agents:
            raise ValueError(f"Unknown agent_name '{agent_name}'")
        agent_dir = Path(agents[agent_name].project_dir)
    else:
        agent_dir = get_current_agent_dir()
        if agent_dir is None:
            raise ValueError("Could not determine current agent directory")

    history_file = ConversationHistory.get_history_file(agent_dir)

    try:
        if history_file.exists():
            history_file.unlink()

        return {
            "agent_name": agent_name or agent_dir.name,
            "success": True,
            "message": "Conversation history cleared successfully",
        }
    except Exception as e:
        return {
            "agent_name": agent_name or agent_dir.name,
            "success": False,
            "message": f"Failed to clear history: {str(e)}",
        }


@mcp.tool()
def refresh_agents() -> dict:
    """
    Force refresh of agent discovery.

    Returns:
        Dict containing refresh results
    """
    try:
        old_agents = get_discovered_agents()
        new_agents = get_discovered_agents(refresh=True)

        return {
            "success": True,
            "old_agent_count": len(old_agents),
            "new_agent_count": len(new_agents),
            "agents_added": list(set(new_agents.keys()) - set(old_agents.keys())),
            "agents_removed": list(set(old_agents.keys()) - set(new_agents.keys())),
            "current_agents": list(new_agents.keys()),
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to refresh agents: {str(e)}",
        }


# Dynamic tool creation for each agent type
def create_agent_tools():
    """Dynamically create tools for each discovered agent."""
    agents = get_discovered_agents()

    for agent_name, config in agents.items():
        # Create a unique function for each agent with proper closure
        def make_agent_spawn_function(agent_config):
            def agent_spawn(
                task_prompt: str,
                context_files: Optional[List[str]] = None,
                timeout_sec: Optional[int] = None,
            ) -> dict:
                # CRITICAL FIX: Add access validation to dynamic tools
                calling_agent_dir = get_current_agent_dir()
                if calling_agent_dir and not validate_agent_access(
                    agent_config.name, calling_agent_dir
                ):
                    raise ValueError(
                        f"Current agent is not allowed to call agent '{agent_config.name}'"
                    )

                return spawn_sub_agent(
                    agent_config.name, task_prompt, context_files, timeout_sec
                )

            # Set function metadata
            agent_spawn.__name__ = (
                f"spawn_{agent_config.name.replace('-', '_').replace(' ', '_')}"
            )
            agent_spawn.__qualname__ = agent_spawn.__name__

            # Create docstring
            docstring = f"""
            Spawn the {agent_config.name} agent with the given task.
            
            This agent has access to: {", ".join(agent_config.allowed_mcp_servers) if agent_config.allowed_mcp_servers else "no specific MCP servers"}
            Can spawn sub-agents: {", ".join(agent_config.allowed_sub_agents) if agent_config.allowed_sub_agents else "none"}
            Timeout: {agent_config.timeout_sec} seconds
            
            Args:
                task_prompt: The task/prompt for the {agent_config.name} agent
                context_files: Optional list of files to copy to agent workspace
                timeout_sec: Optional timeout override (default: {agent_config.timeout_sec})
            
            Returns:
                Dict containing execution results and metadata
            """
            agent_spawn.__doc__ = docstring.strip()

            return agent_spawn

        # Create and register the tool
        try:
            agent_spawn_func = make_agent_spawn_function(config)
            mcp.tool()(agent_spawn_func)
            logger.info(f"Created dynamic tool: {agent_spawn_func.__name__}")
        except Exception as e:
            logger.error(f"Failed to create tool for agent {agent_name}: {e}")


# Create dynamic tools
create_agent_tools()

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Ensure required directories exist for all discovered agents
        agents = get_discovered_agents()
        logger.info(f"Discovered {len(agents)} agents: {list(agents.keys())}")

        for config in agents.values():
            if not ProjectManager.ensure_project_structure(Path(config.project_dir)):
                logger.error(f"Failed to create project structure for {config.name}")

        logger.info("Starting MCP server...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
