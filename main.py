from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig,set_tracing_disabled
import os
from dotenv import load_dotenv
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key=os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model=OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

agent=Agent(
    name="student_Assistant",
    model=model,
    instructions="You are a helpful assistant that answers questions about chainlit-project.",

)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome to the chainlit-project! How can I assist you today?").send()

@cl.on_message
async def handle_on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(
        agent,
        input=history,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    history.append({"role": "assistant", "content": result.final_output})

    await cl.Message(content=result.final_output).send()
    cl.user_session.set("history", history)
