import asyncio
import json
import threading

from aiohttp import web, WSMsgType

from llm.z_ai.client import zai_client

MODEL = 'glm-5.3'
_DONE = object()


def _produce_chunks(system_prompt, user_prompt, loop, queue, stop_event):
    '''
        Runs in a worker thread: iterates the blocking Z.ai stream and hands each delta to the event loop.
    '''
    try:
        response = zai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            stream=True,
        )

        for chunk in response:
            if stop_event.is_set():
                break
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if getattr(delta, 'reasoning_content', None):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "reasoning", "delta": delta.reasoning_content})

            if getattr(delta, 'content', None):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "content", "delta": delta.content})

    except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, _DONE)


async def stream_llm(ws, system_prompt, user_prompt):
    '''
        Streams one LLM response over the websocket, then sends {"type": "done"}.
    '''
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stop_event = threading.Event()
    worker = loop.run_in_executor(None, _produce_chunks, system_prompt, user_prompt, loop, queue, stop_event)

    try:
        while (item := await queue.get()) is not _DONE:
            if ws.closed:
                break
            await ws.send_json(item)

        if not ws.closed:
            await ws.send_json({"type": "done"})
    finally:
        # Stop the worker if the client disconnected mid-stream
        stop_event.set()
        await worker


async def websocket_handler(request):
    '''
        Client sends: {"system_prompt": "...", "user_prompt": "..."}
        Server sends: {"type": "reasoning" | "content", "delta": "..."} ... then {"type": "done"}
                      or {"type": "error", "message": "..."}
    '''
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    print("[WS LOG] client connected")

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
                user_prompt = payload['user_prompt']
            except (json.JSONDecodeError, KeyError, TypeError):
                await ws.send_json({"type": "error",
                                    "message": "Expected JSON with 'user_prompt' (and optional 'system_prompt')"})
                continue

            await stream_llm(ws, payload.get('system_prompt', ''), user_prompt)

        elif msg.type == WSMsgType.ERROR:
            print(f"[WS LOG] connection closed with exception {ws.exception()}")

    print("[WS LOG] client disconnected")
    return ws


async def health_handler(request):
    return web.json_response({"status": "ok", "websocket": "/ws"})


def create_app():
    app = web.Application()
    app.router.add_get('/', health_handler)
    app.router.add_get('/ws', websocket_handler)
    return app


def main():
    web.run_app(create_app(), host='127.0.0.1', port=8080)
