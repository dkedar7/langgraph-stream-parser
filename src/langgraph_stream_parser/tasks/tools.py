"""Agent-facing delegation tools — let an agent spawn async copies of itself.

These mirror the deepagents async-subagent tool contract, but run against the
LOCAL :class:`TaskRunner` + store (no remote Agent Protocol server). The agent
gets a ``task_id`` back immediately and should NOT poll — the task runs in the
background while the agent keeps working.

Wire them into a host's toolset alongside its other tools; they reach the
process-global runner via :func:`get_runner`, so a sub-task the agent spawns is
automatically linked to the spawning task (``parent_id``) via the
``current_task_id`` context var.
"""
from __future__ import annotations

from langchain_core.tools import tool as langchain_tool

from .runner import current_task_id, get_runner

_UNAVAILABLE = "Async task delegation is unavailable in this context."


@langchain_tool
async def start_async_task(title: str, prompt: str) -> str:
    """Delegate a task to a background copy of yourself.

    Returns a task_id immediately; the task runs in the background while you
    keep working. Do NOT wait or poll — report the task_id to the user and move
    on. Use this for long or parallelizable work.

    Args:
        title: A short human-readable name for the task.
        prompt: The full instruction for the background agent to carry out.
    """
    runner = get_runner()
    if runner is None:
        return _UNAVAILABLE
    try:
        task_id = await runner.enqueue(
            title=title, prompt=prompt, parent_id=current_task_id.get()
        )
    except ValueError as e:
        return f"Could not start task: {e}"
    return f"Started async task '{title}'. task_id: {task_id} (running in the background)."


@langchain_tool
async def check_async_task(task_id: str) -> str:
    """Check a delegated task's current status and result. Does not block.

    Call this once when the user asks about a task — never poll in a loop.
    """
    runner = get_runner()
    if runner is None:
        return _UNAVAILABLE
    task = await runner.store.get(task_id)
    if task is None:
        return f"No delegated task with id {task_id}."
    state = task.get("state")
    if state == "done":
        return f"Task {task_id} is done.\n\nResult:\n{task.get('result') or '(no text output)'}"
    if state == "failed":
        return f"Task {task_id} failed: {task.get('error') or 'unknown error'}"
    if state == "review_needed":
        return f"Task {task_id} is paused and needs human review/approval before it can continue."
    return f"Task {task_id} is {state}."


@langchain_tool
async def list_async_tasks() -> str:
    """List delegated tasks and their current states."""
    runner = get_runner()
    if runner is None:
        return _UNAVAILABLE
    tasks = await runner.store.list()
    if not tasks:
        return "No delegated tasks."
    lines = [
        f"- {t['task_id']} [{t.get('state')}] {t.get('title')}" for t in tasks
    ]
    return "Delegated tasks:\n" + "\n".join(lines)


@langchain_tool
async def update_async_task(task_id: str, message: str) -> str:
    """Send a follow-up instruction to a finished delegated task.

    Continues that task's conversation on its own thread (it remembers its prior
    work) and runs in the background. Use for "now also do X" on a completed task.
    """
    runner = get_runner()
    if runner is None:
        return _UNAVAILABLE
    ok = await runner.followup(task_id, message)
    return (
        f"Sent follow-up to task {task_id} (running in the background)."
        if ok
        else f"Could not update task {task_id} (not found, or it is still running)."
    )


@langchain_tool
async def cancel_async_task(task_id: str) -> str:
    """Cancel a delegated task that is queued, running, or awaiting review."""
    runner = get_runner()
    if runner is None:
        return _UNAVAILABLE
    ok = await runner.cancel(task_id)
    return (
        f"Cancelled task {task_id}."
        if ok
        else f"Could not cancel task {task_id} (not found or already finished)."
    )


TASK_TOOLS = [
    start_async_task,
    check_async_task,
    list_async_tasks,
    update_async_task,
    cancel_async_task,
]
