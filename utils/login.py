import os
import streamlit as st
import pandas as pd
from utils.st_auth import auth_basic

LOGS = "./logs"
ADMIN = "mary"


def show_logs():
    logs = [os.path.join(LOGS, f) for f in os.listdir(LOGS)
            if (f.endswith(".csv") and "ai_answers" in f)]
    log_keys = [f.split("_")[-1].replace(".csv", "") for f in logs]

    for log_key, log in zip(log_keys, logs):
        st.markdown(f"#### Лог для ключа :green[**{log_key}**]")
        df = pd.read_csv(log)
        st.dataframe(df, width=1000, use_container_width=True)


@auth_basic
def main():
    is_admin = st.session_state.get("username") == ADMIN
    if is_admin:
        show_logs()
        # logs = [os.path.join(LOGS, f) for f in os.listdir(LOGS)]
        # st.write(logs)
        # for log in logs:
        #     with open(log, "r", encoding="utf-8") as f:
        #         st.write(f.read())
