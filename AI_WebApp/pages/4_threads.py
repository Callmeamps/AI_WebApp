import os
import streamlit as st
import requests
import json
from config import my_keys

nocodb_api_key = os.environ['NOCODB_API_KEY']

convo_api_url = os.environ['convo_api_url']
msg_api_url = os.environ['msg_api_url']

headers = {
  "accept": "application/json",
  "xc-token": nocodb_api_key,
  "Content-Type": "application/json"
}

convo_response = json.loads(requests.get(convo_api_url, headers=headers).text)
chat_history = convo_response["list"]

thrds, chts = st.columns([1, 4])

# create a state variable to store the selected conversation id
selected_convo_id = st.session_state.get("selected_convo_id", None)


def get_msgs(convo_id):
  # update the state variable with the selected conversation id
  st.session_state.selected_convo_id = convo_id


def display_msgs():
  if selected_convo_id is not None:
    # filter messages based on the selected conversation id
    msg_response = json.loads(requests.get(msg_api_url, headers=headers).text)
    msg_history = msg_response["list"]
    msgs = [
      msg for msg in msg_history
      if msg["nc_556n___Convos_id"] == selected_convo_id
    ]

    with chts:
      st.header(chat_history[selected_convo_id]["Title"])
      "---"
      if chat_history[selected_convo_id]["summary"] is None:
        st.write("No summary to display. :unamused:")
      else:
        st.subheader("Chat Summary")
        st.write(chat_history[selected_convo_id]["summary"])
      "---"
      for msg in msgs:
        ucol, mucol = st.columns([1, 4])
        usr = msg["user_message"]
        ai = msg["ai_message"]
        with ucol:
          st.subheader("User")
        with mucol:
          st.write(usr)
        "---"
        aicol, maicol = st.columns([1, 4])
        with aicol:
          st.subheader("AI")
        with maicol:
          st.write(ai)


for i, convo in enumerate(chat_history):
  convo_title = convo["Title"]
  convo_id = convo["Id"]
  with thrds:
    if selected_convo_id == convo_id:
      st.write(f"> **{convo_title}**")
    else:
      if st.button(str(convo_id),
                   on_click=get_msgs,
                   args=(convo_id, ),
                   use_container_width=True):
        selected_convo_id = convo_id
        st.session_state.selected_convo_id = convo_id
      st.write(convo_title)

display_msgs()
