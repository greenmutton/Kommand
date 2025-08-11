# Kommand

An MCP server for spawning isolated Qwen sub-agents with secure project isolation and dynamic tool management.

## Features

- **Dynamic Agent Discovery**: Automatically discovers and loads agent configurations
- **Project Isolation**: Each agent operates in isolated environments with separate workspaces
- **Security-First Design**: Comprehensive file safety checks and environment variable protection
- **Conversation Management**: Persistent history and context sharing between agents
- **Dynamic Tool Generation**: Auto-creates MCP tools for each discovered agent type
- **Flexible Configuration**: Customizable timeouts, cleanup policies, and agent permissions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/greenmutton/kommand.git
   cd kommand
   ```

2. Install dependencies:
   ```bash
   pip install fastmcp==2.11.0
   ```

3. Set up environment variables:
   ```bash
   export KOMMAND_BASE_DIR="$HOME/base/kommand"
   export KOMMAND_VENV="$HOME/base/venvs/kommand/bin/activate"
   ```

## Configuration

Create agent directories under `$KOMMAND_BASE_DIR/agents/` with:
- `specialization.json` - Agent configuration
- `system_prompt.txt` - Agent's system prompt
- `.env` - Environment variables, such as OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

Example `specialization.json`:
```json
{
  "mcp_servers": ["filesystem", "git"],
  "sub_agents": ["researcher", "analyst"],
  "timeout_sec": 30000,
  "cleanup_after": true,
  "max_workspace_files": 50
}
```

## MCP Usage

```json
"mcpServers": {
  "sub-agent": {
    "command": "/home/dex/base/venvs/c2/bin/python",
    "args": [
      "/home/dex/base/cautiontiger/tools/mcp_sub_agent2.py"
    ],
    "env": {
      "KOMMAND_BASE_DIR": "/path/to/base/dir",
      "KOMMAND_VENV": "/path/to/venvs/kommander-venv/bin/activate"
    }
  }
}
```

## Agent Structure

Each agent has:
- `workspace/` - Temporary working files
- `output/` - Generated artifacts
- `logs/` - Execution logs
- `conversation_history.json` - Persistent conversation history

## Security

- File safety checks with extension and content validation
- Sensitive environment variable protection
- Path traversal prevention
- Execution permission restrictions
- Workspace file count and size limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
