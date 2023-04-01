import os
import streamlit as st
from langchain import PromptTemplate, SerpAPIWrapper, SelfAskWithSearchChain
from langchain.llms import OpenAI
import config

#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
search_template = """
I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. I have access to the SerpAPI and I refer to the internet for more accurate answers. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown". 
Q - {query}
A - 
"""

search_prompt = PromptTemplate(template=search_template,
                               input_variables=["query"])


#cohere_llm = Cohere(temperature=0, model="command-xlarge-20221108")
#search = SerpAPIWrapper()
#self_ask_with_search_cohere = SelfAskWithSearchChain(llm=cohere_llm, search_chain=search, verbose=True)

gpt3 = OpenAI(temperature=0)
search = SerpAPIWrapper()
self_ask_with_search_openai = SelfAskWithSearchChain(llm=gpt3,
                                                     search_chain=search,
                                                     verbose=False)

st.set_page_config(page_title="AceTray", page_icon=":alien:")

st.title("SearchGPT")
st.sidebar.title("Main Menu")

st.header("Ace Tray | Mini Amps")
col1, col2 = st.columns(2)

with col1:
  st.markdown("""
  ## Everybody has an assistant nowadays. Why not have my own?
  The first implementation of Ace Tray aka Mini Me.
  This is an active project and will continually go through changes and upgrades.
  
  ### [CallMeAmps](blog.callmeamps.one)
  """)

with col2:
  st.image("cma.jpg", width=768)

st.markdown("### Ask Anything")


def get_text():
  usr_query = st.text_area(label="Your question", placeholder="")
  return usr_query


query_input = get_text()

st.markdown("### Ace Tray's Answer")

if query_input:

  search_w_format = search_prompt.format(query=query_input)

  search_res = self_ask_with_search_openai.run(search_w_format)

  st.write(search_res)
