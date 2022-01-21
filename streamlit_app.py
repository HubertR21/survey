import streamlit as st

from helpers.drive.api import GSAuthentication


st.set_page_config(layout="wide")
st.sidebar.markdown('## Authorization')
user_id = st.sidebar.text_input('Your ID')
user_password = st.sidebar.text_input('Password', type="password")

st.session_state["gsheet"] = GSAuthentication()