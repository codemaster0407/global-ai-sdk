from llm.z_ai.client import zai_client
from llm.z_ai.tools import TOOLS
from pydantic import BaseModel, create_model
from typing import Any
import inspect
import json


def llm_response(model = 'glm-5.3', system_prompt = '', user_prompt = ''):
    '''
        Generate the LLM response. 
    '''
    response = zai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt ,
            },
        ],
        thinking={
            "type": "enabled",  # Optional: "enabled" only, reasoning is always enabled
            "clearThinking" : True
        },
        reasoning_effort="max",
        max_tokens=4096,
        temperature=0.1,
    )

    return response.choices[0].message 



def llm_structured_response(output_model: type[BaseModel], model = 'glm-5.3', system_prompt = '', user_prompt = ''):
    '''
        Generate the LLM response parsed into `output_model` (a pydantic BaseModel subclass).
    '''
    schema_prompt = (
        f'{system_prompt}\n\n'
        'Respond ONLY with a JSON object that matches this JSON schema:\n'
        f'{json.dumps(output_model.model_json_schema())}'
    )

    response = zai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": schema_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=4096,
        temperature=0.1,
    )

    return output_model.model_validate_json(response.choices[0].message.content)


def streaming_response(system_prompt, user_prompt ):

    response = zai_client.chat.completions.create(
        model="glm-5.3",
        messages=[{"role": "system", "content": system_prompt},
                  {"role" : "user" , "content" : user_prompt}],
        stream=True,
        tool_stream=True,
    )

    # Initialize streaming collection variables
    reasoning_content = ""
    content = ""
    final_tool_calls = {}
    reasoning_started = False
    content_started = False

    # Process streaming response
    for chunk in response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Streaming reasoning process output
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not reasoning_started and delta.reasoning_content.strip():
                print("\n🧠 Thinking Process:")
                reasoning_started = True
            reasoning_content += delta.reasoning_content
            print(delta.reasoning_content, end="", flush=True)

        # Streaming answer content output
        if hasattr(delta, 'content') and delta.content:
            if not content_started and delta.content.strip():
                print("\n\n💬 Answer Content:")
                content_started = True
            content += delta.content
            print(delta.content, end="", flush=True)

        # Streaming tool call information (parameter concatenation)
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index
                if idx not in final_tool_calls:
                    final_tool_calls[idx] = tool_call
                    final_tool_calls[idx].function.arguments = tool_call.function.arguments
                else:
                    final_tool_calls[idx].function.arguments += tool_call.function.arguments

    # Output final tool call information
    if final_tool_calls:
        print("\n📋 Function Calls Triggered:")
        for idx, tool_call in final_tool_calls.items():
            print(f"  {idx}: Function Name: {tool_call.function.name}, Parameters: {tool_call.function.arguments}")



def _build_tool(func):
    '''
        Builds the Z.ai tool schema and an argument validator from a function's signature and docstring.
    '''
    fields = {
        name: (Any if param.annotation is inspect.Parameter.empty else param.annotation,
               ... if param.default is inspect.Parameter.empty else param.default)
        for name, param in inspect.signature(func).parameters.items()
    }
    print(fields)
    args_model = create_model(f'{func.__name__}_args', **fields)
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or '',
            "parameters": args_model.model_json_schema(),
        },
    }
    return schema, args_model


_TOOL_REGISTRY = {func.__name__: (func, *_build_tool(func)) for func in TOOLS}


def _run_tool(tool_call):
    '''
        Executes one tool call. Errors are returned to the LLM as the tool result so it can recover.
    '''
    name = tool_call.function.name
    if name not in _TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})

    func, _, args_model = _TOOL_REGISTRY[name]
    try:
        args = args_model.model_validate_json(tool_call.function.arguments or '{}')
        return json.dumps(func(**dict(args)), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def inference_with_tools(system_prompt, user_prompt, model = 'glm-5.3', max_steps = 5):
    '''
        Generate the LLM response, running any tools from tools.py the LLM calls until it gives a final answer.
    '''
    messages = [{"role": "system", "content": system_prompt},
                {"role" : "user" , "content" : user_prompt}]
    tools = [schema for _, schema, _ in _TOOL_REGISTRY.values()]

    for _ in range(max_steps):
        response = zai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            result = _run_tool(tool_call)
            print(f"[TOOL LOG] {tool_call.function.name}({tool_call.function.arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    raise RuntimeError(f"No final answer after {max_steps} tool-calling steps")
