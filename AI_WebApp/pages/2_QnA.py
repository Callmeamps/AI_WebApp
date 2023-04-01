import streamlit as st
from streamlit_chat import message as st_message
#from langchain.chat_models import
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
#from langchain.memory import ConversationBufferWindowMemory
#from langchain.prompts.chat import (
#  ChatPromptTemplate,
#  SystemMessagePromptTemplate,
#  AIMessagePromptTemplate,
#  HumanMessagePromptTemplate,
#)

chatgpt = OpenAI(temperature=0)

template = """You are a helpful and highly intelligent AI Assistant called 'MMACIA' Multi-Model & Agent Chain Intergrated Assistant, you are witty, creative, clever, and very friendly. If you asked a question that is nonsense, trickery, or has no clear answer, respond with 'Unknown'

MMACIA is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, MMACIA is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

MMACIA is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, MMACIA is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, MMACIA is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, MMACIA is here to assist.

Human: {human_input}
MMACIA:"""
chat_prompt = PromptTemplate(input_variables=["human_input"],
                             template=template)

st.set_page_config(page_title="ChatGPT")
st.title("ChatGPT")
st.sidebar.success("Main Menu")

if "history" not in st.session_state:
  st.session_state.history = []

chatgpt_chain = LLMChain(
  llm=chatgpt,
  prompt=chat_prompt,
  verbose=False,
)


def gen_convo():

  user_message = st.session_state.usr_msg
  bot_message = chatgpt_chain.predict(human_input=user_message)

  st.session_state.history.append({"message": user_message, "is_user": True})
  st.session_state.history.append({"message": bot_message, "is_user": False})


st.text_area("Ask MMACIA anything!", key="usr_msg", on_change=gen_convo)

for chat in st.session_state.history:
  st_message(**chat)
