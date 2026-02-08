"""
FastAPI WebSocket Example for LangGraph Stream Parser.

This example demonstrates how to stream LangGraph agent events
to a web client via WebSockets. Events are parsed and sent as
JSON messages that can be rendered in a frontend UI.

Requirements:
    pip install fastapi uvicorn websockets langgraph langchain-openai

Run:
    uvicorn examples.fastapi_websocket:app --reload

Then open http://localhost:8000 in your browser.
"""
import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from langgraph_stream_parser import StreamParser, event_to_dict
from langgraph_stream_parser.events import InterruptEvent
from dotenv import load_dotenv
load_dotenv()

from .agent import agent


app = FastAPI()


async def stream_and_send(
    websocket: WebSocket,
    input_data: Any,
    config: dict,
    parser: StreamParser,
) -> InterruptEvent | None:
    """Stream agent events to websocket, return interrupt if one occurs."""
    loop = asyncio.get_event_loop()

    def stream_events():
        stream = agent.stream(
            input_data,
            config=config,
            stream_mode="updates",
        )
        return list(parser.parse(stream, stream_mode="updates"))

    events = await loop.run_in_executor(None, stream_events)

    for event in events:
        await websocket.send_json(event_to_dict(event))
        await asyncio.sleep(0.01)

        # Return interrupt for handling
        if isinstance(event, InterruptEvent):
            return event

    return None


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with the agent."""
    await websocket.accept()

    # Create a unique thread ID for this connection
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    parser = StreamParser()

    # Track pending interrupt for this session
    pending_interrupt: InterruptEvent | None = None

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle interrupt decision from client
            if message.get("type") == "decision":
                if pending_interrupt is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": "No pending interrupt to respond to",
                    })
                    continue

                decision_type = message.get("decision", "reject")

                # Build resume input using the InterruptEvent helper
                args_modifier = None
                if decision_type == "edit" and "args" in message:
                    edited_args = message["args"]
                    args_modifier = lambda _: edited_args

                resume_input = pending_interrupt.create_resume(
                    decision_type, args_modifier=args_modifier
                )
                pending_interrupt = None

                await websocket.send_json({
                    "type": "decision_ack",
                    "decision": decision_type,
                })

                # Continue streaming from resume
                interrupt = await stream_and_send(
                    websocket, resume_input, config, parser
                )
                if interrupt:
                    pending_interrupt = interrupt

                continue

            # Handle regular chat message
            user_input = message.get("message", "")
            if not user_input:
                continue

            # Send acknowledgment
            await websocket.send_json({
                "type": "user_message",
                "content": user_input,
            })

            # Stream agent response
            input_data = {"messages": [("user", user_input)]}
            interrupt = await stream_and_send(
                websocket, input_data, config, parser
            )
            if interrupt:
                pending_interrupt = interrupt

    except WebSocketDisconnect:
        print(f"Client disconnected (thread: {thread_id})")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e),
        })


# Simple HTML client for testing
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Stream Parser - WebSocket Demo</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        #chat {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            height: 500px;
            overflow-y: auto;
            padding: 16px;
            margin-bottom: 16px;
        }
        .message {
            margin: 8px 0;
            padding: 12px;
            border-radius: 8px;
        }
        .user {
            background: #007bff;
            color: white;
            margin-left: 20%;
        }
        .assistant {
            background: #e9ecef;
            margin-right: 20%;
        }
        .tool-start {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            font-size: 0.9em;
        }
        .tool-end {
            background: #d4edda;
            border-left: 4px solid #28a745;
            font-size: 0.9em;
        }
        .tool-error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        .complete {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            text-align: center;
            font-style: italic;
        }
        .error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        .interrupt {
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 16px;
        }
        .interrupt h4 {
            margin: 0 0 12px 0;
            color: #856404;
        }
        .interrupt pre {
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.85em;
        }
        .interrupt-buttons {
            margin-top: 12px;
            display: flex;
            gap: 8px;
        }
        .interrupt-buttons button {
            padding: 8px 16px;
            font-size: 14px;
        }
        .btn-approve {
            background: #28a745;
        }
        .btn-approve:hover {
            background: #218838;
        }
        .btn-reject {
            background: #dc3545;
        }
        .btn-reject:hover {
            background: #c82333;
        }
        .decision-ack {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            font-style: italic;
        }
        #input-area {
            display: flex;
            gap: 8px;
        }
        #message-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .typing {
            color: #666;
            font-style: italic;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
        }
    </style>
</head>
<body>
    <h1>LangGraph Stream Parser Demo</h1>
    <p>Try asking about the weather or doing calculations!</p>

    <div id="chat"></div>

    <div id="input-area">
        <input type="text" id="message-input" placeholder="Type a message..." />
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');

        let ws = null;
        let currentAssistantMessage = null;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws/chat`);

            ws.onopen = () => {
                addMessage('system', 'Connected to agent.');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };

            ws.onclose = () => {
                addMessage('error', 'Disconnected from server. Refresh to reconnect.');
                sendBtn.disabled = true;
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        function handleEvent(data) {
            switch (data.type) {
                case 'user_message':
                    addMessage('user', data.content);
                    currentAssistantMessage = null;
                    break;

                case 'content':
                    if (!currentAssistantMessage) {
                        currentAssistantMessage = addMessage('assistant', '');
                    }
                    currentAssistantMessage.textContent += data.content;
                    break;

                case 'tool_start':
                    addMessage('tool-start',
                        `🔧 Calling <code>${data.name}</code> with: ${JSON.stringify(data.args)}`);
                    break;

                case 'tool_end':
                    const cls = data.status === 'success' ? 'tool-end' : 'tool-error';
                    const icon = data.status === 'success' ? '✓' : '✗';
                    addMessage(cls, `${icon} <code>${data.name}</code>: ${data.result}`);
                    break;

                case 'complete':
                    addMessage('complete', '— Response complete —');
                    sendBtn.disabled = false;
                    break;

                case 'error':
                    addMessage('error', `Error: ${data.error}`);
                    sendBtn.disabled = false;
                    break;

                case 'interrupt':
                    showInterrupt(data);
                    break;

                case 'decision_ack':
                    addMessage('decision-ack', `Decision: ${data.decision}`);
                    break;
            }

            chat.scrollTop = chat.scrollHeight;
        }

        function showInterrupt(data) {
            const div = document.createElement('div');
            div.className = 'message interrupt';

            let actionsHtml = '';
            for (const action of data.action_requests) {
                actionsHtml += `<p><strong>Tool:</strong> <code>${action.tool || action.action?.tool || 'unknown'}</code></p>`;
                const args = action.args || action.action?.args || {};
                actionsHtml += `<pre>${JSON.stringify(args, null, 2)}</pre>`;
            }

            // Get allowed decisions (now included directly in event)
            const allowedDecisions = data.allowed_decisions || ['approve', 'reject'];

            let buttonsHtml = '<div class="interrupt-buttons">';
            if (allowedDecisions.includes('approve')) {
                buttonsHtml += `<button class="btn-approve" onclick="sendDecision('approve')">Approve</button>`;
            }
            if (allowedDecisions.includes('reject')) {
                buttonsHtml += `<button class="btn-reject" onclick="sendDecision('reject')">Reject</button>`;
            }
            buttonsHtml += '</div>';

            div.innerHTML = `
                <h4>Action Required</h4>
                ${actionsHtml}
                ${buttonsHtml}
            `;

            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function sendDecision(decision) {
            if (!ws) return;

            // Remove interrupt buttons after decision
            const interruptBtns = document.querySelectorAll('.interrupt-buttons');
            interruptBtns.forEach(el => el.remove());

            ws.send(JSON.stringify({ type: 'decision', decision }));
        }

        function addMessage(type, content) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function sendMessage() {
            const message = input.value.trim();
            if (!message || !ws) return;

            ws.send(JSON.stringify({ message }));
            input.value = '';
            sendBtn.disabled = true;
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        connect();
    </script>
</body>
</html>
"""


@app.get("/")
async def get_client():
    """Serve the HTML test client."""
    return HTMLResponse(HTML_CLIENT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
