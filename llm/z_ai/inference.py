from llm.z_ai.client import zai_client
from pydantic import BaseModel
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