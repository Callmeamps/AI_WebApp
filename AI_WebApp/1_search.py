import os
import gradio as gr
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

gpt3 = OpenAI(temperature=0, model="text-davinci-002")
search = SerpAPIWrapper()
self_ask_with_search_openai = SelfAskWithSearchChain(llm=gpt3,
                                                     search_chain=search,
                                                     verbose=True)

with gr.Blocks() as demo:
    gr.Markdown("# SearchGPT")
    with gr.Box():
        with gr.Column():
            gr.Markdown("""
            ## Everybody has an assistant nowadays. Why not have my own?
            The first implementation of Ace Tray aka Mini Me.
            This is an active project and will continually go through changes and upgrades.
            
            ### [CallMeAmps](blog.callmeamps.one)
            """)

        with gr.Column():
            gr.Image(value="cma.jpg")

        usr_query = gr.TextArea(label="Your question", placeholder="Ask me anything.")


        send_btn = gr.Button("Send")

        gr.Markdown("### Ace Tray's Answer")

        formatted_search = search_prompt.format(query=usr_query)

        def search_res():
            response = self_ask_with_search_openai.run(formatted_search)
            return response

        results = gr.TextArea()
        send_btn.click(fn=search_res, outputs=results)

demo.launch()