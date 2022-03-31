import streamlit as st


def update_point_color(df_incomplete, incomplete_column):
    df_incomplete.at[st.session_state['current_null_index'], incomplete_column] = st.session_state['my_slider']

def update_session_state(X, y_pred, border_points):
    st.session_state.X = X
    st.session_state['border_points'] = border_points
    st.session_state['y_pred'] = y_pred
