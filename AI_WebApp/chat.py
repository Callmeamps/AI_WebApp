import json
import os
from datetime import datetime
import gradio as gr
import requests
import config
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  #  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
os.environ["LANGCHAIN_HANDLER"] = "langchain"
# nocodb_api_key = os.environ['NOCODB_API_KEY']
# openai_api_key = os.environ['OPENAI_API_KEY']

chatgpt = ChatOpenAI(temperature=0)
gpt3 = OpenAI(temperature=0) 
davinci_002 = OpenAI(temperature=0, model_name="text-davinci-002")  #change to text-davinci-002

system_template = """You are a helpful and highly intelligent AI Assistant called 'MMACIA' Multi-Model & Agent Chain Intergrated Assistant, you are witty, creative, clever, and very friendly. If you asked a question that is nonsense, trickery, or has no clear answer, respond with 'Unknown'

MMACIA is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, MMACIA is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

MMACIA is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, MMACIA is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, MMACIA is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, MMACIA is here to assist.
"""
title_template = """Generate a short and concise title for the following conversation:
{convo_history}
"""
summary_template = """Generate a short and concise executive summary for the following conversation:
{convo_history}
"""

title_prompt = PromptTemplate(input_variables=["convo_history"],
                              template=title_template)

summary_prompt = PromptTemplate(input_variables=["convo_history"],
                                template=summary_template)

title_chain = LLMChain(
  llm=gpt3,
  prompt=title_prompt,
  verbose=False,
)

summary_chain = LLMChain(
  llm=gpt3,
  prompt=summary_prompt,
  verbose=False,
)

system_message_prompt = SystemMessagePromptTemplate.from_template(
  system_template)
human_template = "{human_input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
  [system_message_prompt, human_message_prompt])

memory = ConversationBufferWindowMemory(memory_key="chat_history")


chatgpt_chain = LLMChain(
  llm=chatgpt,
  prompt=chat_prompt,
  verbose=True,
  memory=memory
)

basic_tools = load_tools(["serpapi", "wolfram-alpha", "requests_all", "wikipedia"], llm=chatgpt)

agent = initialize_agent(basic_tools, chatgpt, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

def ChatAgent(user_input: str, chat_history: list) -> str:
    response = agent({"input": user_input, "chat_history": chat_history})
    res = json.dumps(response["intermediate_steps"], indent=2)
    return res

# def RunApp():
#     chat_history = []
#     while True:
#         user_input = input(f"{chat_history[0],}:\n")
#         if user_input.lower() == "exit":
#             break
#         agent_res = ChatAgent(user_input=user_input, chat_history=chat_history)
#         chat_history.append({"role": "user", "content": user_input})
#         print(agent_res)
#         chat_history.append({"role": "assistant", "content": agent_res})

# RunApp()

chat_history = []


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history, message):
    response = chatgpt_chain.run(human_input=message)
    history[-1][1] = response
    return history

def RunAgent(history, message):
    response = ChatAgent(user_input=message, chat_history=history)
    history[-1][1] = response
    return history

with gr.Blocks() as demo:
    gr.Markdown("# AI Chat")
    with gr.Box():
        with gr.Column():
            gr.Markdown("""
            ## Everybody has an assistant nowadays. Why not have my own?
            The first implementation of an AI chatbot.
            This is an active project and will continually go through changes and upgrades.
            
            ### [CallMeAmps](blog.callmeamps.one)
            """)

        with gr.Column():
            gr.Image(value="cma.jpg")

    chatbot = gr.Chatbot(chat_history, elem_id="chatbot").style(height=750)
    # user_message = st.session_state.usr_msg
    # bot_message = chatgpt_chain.run(human_input=user_message)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image"])
        with gr.Column(scale=0.15, min_width=0):
            sendIt = gr.Button(">>")
            
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, [chatbot, txt], chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(
        bot, [chatbot, txt], chatbot
    )
    sendIt.click(add_text, [chatbot, txt], [chatbot]).then(
        RunAgent, [chatbot, txt], chatbot
    )

demo.launch()