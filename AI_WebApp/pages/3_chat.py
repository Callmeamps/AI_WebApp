import os
import streamlit as st
import requests
import json
from config import my_keys
from datetime import datetime
from streamlit_chat import message as st_message
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate, OpenAI
#from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  #  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)

nocodb_api_key = os.environ['NOCODB_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
chatgpt = ChatOpenAI(temperature=0)

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
gpt3 = OpenAI(temperature=0)  #change to Currie

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

chatgpt_chain = LLMChain(
  llm=chatgpt,
  prompt=chat_prompt,
  verbose=False,
)

st.set_page_config(page_title="ChatGPT")
st.title("ChatGPT")
st.sidebar.header("Threads")

if "history" not in st.session_state:
  st.session_state.history = []

now = datetime.now()
convo_api_url = "https://noco.dev.hunnid.one/api/v1/db/data/v1/langcone_ai/Convos"
msg_api_url = "https://noco.dev.hunnid.one/api/v1/db/data/v1/langcone_ai/messages"

headers = {
  "accept": "application/json",
  "xc-token": nocodb_api_key,
  "Content-Type": "application/json"
}

if "id" not in st.session_state:
  st.session_state.id = 0

if "con_id" not in st.session_state:
  st.session_state.con_id = 0


def gen_convo():
  #global id

  user_message = st.session_state.usr_msg
  bot_message = chatgpt_chain.run(human_input=user_message)

  #convo = {"Id": 0, "Title": "Test1", "system_message": system_template}

  msgs = {
    "Id": st.session_state.id,
    "user_message": user_message,
    "nc_556n___Convos_id": st.session_state.con_id - 1,
    "ai_message": bot_message
  }

  st.session_state.history.append({"message": user_message, "is_user": True})
  st.session_state.history.append({"message": bot_message, "is_user": False})
  #requests.post(convo_api_url, json=convo, headers=headers)
  requests.post(msg_api_url, json=msgs, headers=headers)
  st.session_state.id += 1


convo_response = json.loads(requests.get(convo_api_url, headers=headers).text)
chat_history = convo_response["list"]
for convo in chat_history:
  convo_titles = convo["Title"]
  st.sidebar.write(convo_titles)


def new_chat():
  convo_title = title_chain.run(convo_history=st.session_state.history)
  convo_summary = summary_chain.run(convo_history=st.session_state.history)
  convo_json = {
    "Id": st.session_state.con_id,
    "Title": convo_title,
    "system_message": system_template,
    "summary": convo_summary
  }
  requests.post(convo_api_url, json=convo_json, headers=headers)
  #for chat in st.session_state.history:
  #  del st.session_state[chat]
  if "history" in st.session_state:
    st.session_state.history = []
  #if "con_id" not in st.session_state:
  st.session_state.con_id += 1
  if st.session_state.id > 0:
    st.session_state.id = 0


prmt, snd = st.columns([4, 1])

with prmt:
  st.text_area("Ask me anything!", key="usr_msg", on_change=gen_convo)

with snd:
  st.button("Send", on_click=gen_convo)

st.button("New Chat", on_click=new_chat, use_container_width=True)
c_id, m_id = st.columns(2)
with m_id:
  st.write("current message ID: " + str(st.session_state.id))
with c_id:
  st.write("current conversation ID: " + str(st.session_state.con_id))

for chat in st.session_state.history:
  st_message(**chat)
