import os
import subprocess

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver

# Add a think tool
def think_tool(reflection: str) -> str:
    """A tool to reflect on your actions and reasoning.

    This tool allows you to pause and think about your next steps,
    evaluate your current state, or reconsider your approach. Use 
    this tool to generate internal reflections that the user can see.

    Args:
        reflection: The reflection text
    Returns:
        str: The recorded reflection
    """
    return reflection


def bash(command: str, timeout: int = 60) -> dict:
    """Execute a bash command and return the output.

    Runs the command in the workspace directory. Use this for file operations,
    git commands, installing packages, or any shell operations.

    In virtual filesystem mode (Linux only), commands run in a bubblewrap sandbox
    with network disabled for security.

    Args:
        command: The bash command to execute
        timeout: Maximum time in seconds to wait for the command (default: 60)

    Returns:
        Dictionary containing:
        - stdout: Standard output from the command
        - stderr: Standard error output
        - return_code: Exit code (0 typically means success)
        - status: "success" or "error"

    Examples:
        # List files
        bash("ls -la")

        # Check git status
        bash("git status")

        # Install a package
        bash("pip install pandas")

        # Run a script
        bash("python script.py")
    """

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "status": "success" if result.returncode == 0 else "error"
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "return_code": -1,
            "status": "error"
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "status": "error"
        }

backend = FilesystemBackend(root_dir=os.getcwd(), virtual_mode=True)
agent = create_deep_agent(
    name="Example",
    backend=backend,
    tools=[think_tool, bash],
    interrupt_on=dict(bash=True),
    checkpointer=InMemorySaver()
)