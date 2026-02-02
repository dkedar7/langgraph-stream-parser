"""
Example: Using JupyterDisplay for rich live streaming in notebooks.

This example demonstrates how to use the JupyterDisplay adapter to show
real-time, visually polished output from LangGraph agents in Jupyter notebooks.

To run this example in a Jupyter notebook:
1. Install dependencies: pip install langgraph-stream-parser[jupyter] langgraph langchain-openai
2. Set your OPENAI_API_KEY environment variable
3. Copy this code into a notebook cell and run it
"""

# %% [markdown]
# # LangGraph Stream Parser - Jupyter Display Example
#
# This notebook demonstrates the JupyterDisplay adapter for rich live streaming.

# %% Setup
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

# %% [markdown]
# ## Basic Usage
#
# The simplest way to use JupyterDisplay is with the `stream()` method:

# %%
# Create a display instance
display = JupyterDisplay()

# Stream directly from your graph (uncomment when you have a real graph)
# display.stream(graph.stream({"messages": [("user", "Hello!")]}, stream_mode="updates"))

# %% [markdown]
# ## Manual Control
#
# For more control, you can update the display event by event:

# %%
from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    CompleteEvent,
)

# Create display and parser
display = JupyterDisplay()
parser = StreamParser()

# Example: Manual event processing
# for event in parser.parse(graph.stream(...)):
#     display.update(event)
#     # You can also do custom processing here

# %% [markdown]
# ## Simulated Demo
#
# Here's a simulation showing what the display looks like with various events:

# %%
import time

def simulate_stream():
    """Simulate a stream of events for demonstration."""
    display = JupyterDisplay()
    display.reset()

    # Simulate content streaming
    content_chunks = [
        "Let me ",
        "search for ",
        "that information ",
        "for you.\n\n",
    ]

    for chunk in content_chunks:
        display._process_event(ContentEvent(content=chunk))
        display._render()
        time.sleep(0.3)

    # Simulate tool call start
    display._process_event(ToolCallStartEvent(
        id="call_1",
        name="web_search",
        args={"query": "latest Python release"}
    ))
    display._render()
    time.sleep(1.0)

    # Simulate tool call end
    display._process_event(ToolCallEndEvent(
        id="call_1",
        name="web_search",
        result="Python 3.12 was released in October 2023...",
        status="success"
    ))
    display._render()
    time.sleep(0.5)

    # More content
    more_content = [
        "Based on my search, ",
        "Python 3.12 is the latest stable release. ",
        "It includes several performance improvements ",
        "and new features like improved error messages.",
    ]

    for chunk in more_content:
        display._process_event(ContentEvent(content=chunk))
        display._render()
        time.sleep(0.2)

    # Complete
    display._process_event(CompleteEvent())
    display._render()

# Uncomment to run the simulation:
# simulate_stream()

# %% [markdown]
# ## With Tool Extractions
#
# The display shows extracted data from special tools like `think_tool`:

# %%
def simulate_with_extraction():
    """Simulate stream with tool extraction."""
    display = JupyterDisplay()
    display.reset()

    # Content
    display._process_event(ContentEvent(content="Let me think about this problem...\n"))
    display._render()
    time.sleep(0.5)

    # Think tool
    display._process_event(ToolCallStartEvent(
        id="call_think",
        name="think_tool",
        args={"reflection": "Analyzing the user's request..."}
    ))
    display._render()
    time.sleep(0.8)

    display._process_event(ToolCallEndEvent(
        id="call_think",
        name="think_tool",
        result="Reflection complete",
        status="success"
    ))

    display._process_event(ToolExtractedEvent(
        tool_name="think_tool",
        extracted_type="reflection",
        data="The user wants to understand how to use the streaming parser. I should provide clear examples with both basic and advanced usage patterns."
    ))
    display._render()
    time.sleep(0.5)

    # Continue with response
    display._process_event(ContentEvent(
        content="\n\nHere's how you can use the streaming parser effectively..."
    ))
    display._render()

    display._process_event(CompleteEvent())
    display._render()

# Uncomment to run:
# simulate_with_extraction()

# %% [markdown]
# ## Handling Interrupts
#
# The display prominently shows human-in-the-loop interrupts:

# %%
def simulate_interrupt():
    """Simulate stream with interrupt."""
    display = JupyterDisplay()
    display.reset()

    display._process_event(ContentEvent(
        content="I need to run a command to check the system status.\n"
    ))
    display._render()
    time.sleep(0.5)

    display._process_event(ToolCallStartEvent(
        id="call_bash",
        name="bash",
        args={"command": "rm -rf /tmp/cache/*"}
    ))
    display._render()
    time.sleep(0.3)

    # Interrupt!
    display._process_event(InterruptEvent(
        action_requests=[
            {
                "tool": "bash",
                "tool_call_id": "call_bash",
                "args": {"command": "rm -rf /tmp/cache/*"}
            }
        ],
        review_configs=[
            {"allowed_decisions": ["approve", "reject", "edit"]}
        ]
    ))
    display._render()

# Uncomment to run:
# simulate_interrupt()

# %% [markdown]
# ## Configuration Options

# %%
# Display with custom options
display = JupyterDisplay(
    show_timestamps=True,      # Show timestamps on events
    show_tool_args=True,       # Show tool arguments in status table
    max_content_preview=200,   # Max chars for extraction previews
)

# %% [markdown]
# ## Full Example with Real LangGraph Agent
#
# Here's a complete example with a real LangGraph agent:

# %%
"""
# Uncomment and run this with a real LangGraph setup:

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    '''Get the weather for a city.'''
    return f"The weather in {city} is sunny and 72°F"

@tool
def search_web(query: str) -> str:
    '''Search the web for information.'''
    return f"Search results for '{query}': ..."

# Create agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools=[get_weather, search_web])

# Create display
display = JupyterDisplay()

# Stream with live display
display.stream(
    agent.stream(
        {"messages": [("user", "What's the weather like in San Francisco?")]},
        stream_mode="updates"
    )
)
"""
print("See the commented code above for a complete example with a real agent.")
