"""
Interrupt parsing utilities.

Handles the various interrupt formats that LangGraph can produce
during human-in-the-loop workflows.
"""
from typing import Any


def _extract_from_interrupt_obj(obj: Any) -> tuple[list[Any], list[Any]]:
    """Extract action_requests and review_configs from a single interrupt object.

    Handles both Interrupt objects (with .value dict) and plain dicts/objects.
    """
    if hasattr(obj, 'value') and isinstance(obj.value, dict):
        return (
            obj.value.get('action_requests', []),
            obj.value.get('review_configs', []),
        )
    if isinstance(obj, dict):
        return (
            obj.get('action_requests', []),
            obj.get('review_configs', []),
        )
    return (
        getattr(obj, 'action_requests', []),
        getattr(obj, 'review_configs', []),
    )


def parse_interrupt_value(interrupt_value: Any) -> tuple[list[Any], list[Any]]:
    """Parse interrupt value into action_requests and review_configs.

    Handles different interrupt value formats from LangGraph:
        - Tuple of Interrupt objects (one or more)
        - Object formats with attributes
        - Dict formats with keys

    Args:
        interrupt_value: The interrupt value from LangGraph.
            This is typically found in update["__interrupt__"].

    Returns:
        Tuple of (action_requests, review_configs).
    """
    action_requests: list[Any] = []
    review_configs: list[Any] = []

    if isinstance(interrupt_value, tuple):
        # Check if elements are Interrupt objects (have .value attribute)
        # This handles tuples of any length from LangGraph
        if len(interrupt_value) > 0 and hasattr(interrupt_value[0], 'value'):
            # Tuple of Interrupt objects — aggregate from all
            for interrupt_obj in interrupt_value:
                actions, configs = _extract_from_interrupt_obj(interrupt_obj)
                action_requests.extend(actions)
                review_configs.extend(configs)
        elif len(interrupt_value) == 1:
            # Single-element tuple containing a dict or other object
            action_requests, review_configs = _extract_from_interrupt_obj(
                interrupt_value[0]
            )
        elif len(interrupt_value) == 2:
            # Legacy format: (action_requests, review_configs) as plain lists
            first, second = interrupt_value
            if isinstance(first, list) and isinstance(second, list):
                action_requests, review_configs = first, second
            else:
                # Unknown format — try to extract from each element
                for item in interrupt_value:
                    actions, configs = _extract_from_interrupt_obj(item)
                    action_requests.extend(actions)
                    review_configs.extend(configs)
    elif isinstance(interrupt_value, dict):
        action_requests = interrupt_value.get('action_requests', [])
        review_configs = interrupt_value.get('review_configs', [])
    else:
        # Handle object format
        action_requests = getattr(interrupt_value, 'action_requests', [])
        review_configs = getattr(interrupt_value, 'review_configs', [])

    return action_requests, review_configs


def serialize_action_request(action: Any, index: int) -> dict[str, Any]:
    """Serialize an action request to a dictionary.

    Handles both dict and object formats, and both 'name' and 'tool' field names.

    Args:
        action: The action request object or dict.
        index: The index of this action (used for fallback tool_call_id).

    Returns:
        Dictionary with tool, tool_call_id, args, and description.
    """
    if isinstance(action, dict):
        tool_name = action.get('tool') or action.get('name')
        tool_call_id = action.get('tool_call_id', f"call_{index}")
        args = action.get('args', {})
        description = action.get('description')
    else:
        tool_name = getattr(action, 'tool', None) or getattr(action, 'name', None)
        tool_call_id = getattr(action, 'tool_call_id', f"call_{index}")
        args = getattr(action, 'args', {})
        description = getattr(action, 'description', None)

    return {
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "args": args,
        "description": description,
    }


def serialize_review_config(config: Any) -> dict[str, Any]:
    """Serialize a review config to a dictionary.

    Args:
        config: The review config object or dict.

    Returns:
        Dictionary with allowed_decisions.
    """
    if isinstance(config, dict):
        allowed_decisions = config.get('allowed_decisions', [])
    else:
        allowed_decisions = getattr(config, 'allowed_decisions', [])

    return {
        "allowed_decisions": allowed_decisions,
    }


def process_interrupt(interrupt_value: Any) -> dict[str, Any]:
    """Process a LangGraph interrupt value and convert to serializable format.

    This is the main entry point for interrupt processing. It takes the
    raw interrupt value and produces a normalized dictionary with
    action_requests and review_configs.

    Args:
        interrupt_value: The interrupt value from the update.
            Typically update["__interrupt__"].

    Returns:
        Dictionary containing 'action_requests' and 'review_configs' lists.
    """
    action_requests, review_configs = parse_interrupt_value(interrupt_value)

    interrupt_data: dict[str, Any] = {
        "action_requests": [],
        "review_configs": [],
    }

    # Extract action requests
    for i, action in enumerate(action_requests):
        interrupt_data["action_requests"].append(
            serialize_action_request(action, i)
        )

    # Extract review configs
    for config in review_configs:
        interrupt_data["review_configs"].append(
            serialize_review_config(config)
        )

    return interrupt_data
