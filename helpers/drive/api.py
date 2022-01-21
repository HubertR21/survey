import time
import pandas as pd
import gspread
import streamlit as st
from google.oauth2 import service_account
# from gsheetsdb import connect

class GSAuthentication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GSAuthentication, cls).__new__(cls)
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                ],
            )
            print("created")
            # TODO: refactor to use SQL not custom code
            # conn = connect(credentials=credentials)
            cls.gc = gspread.Client(credentials)
            cls.url = st.secrets['private_gsheets_url']

        return cls._instance


# TODO: remove mock
def get_registered_ids():
    return ["1", "2"]


def save_results(g, y_pred_uncertain, annotator_id, seed, dataset_name, na_fraction):
    gsheet = g.gc.open_by_url(g.url)
    datasets = {"iris": 0, "wola": 1, "stamp_type": 2}
    worksheet_number = datasets[dataset_name]
    wsheet = gsheet.get_worksheet(worksheet_number)
    data = pd.DataFrame(gsheet.sheet1.get_all_values())
    now = time.localtime(time.time())
    row_data = [
        time.strftime("%m/%d/%Y, %H:%M:%S", now),
        annotator_id,
        seed,
        na_fraction,
        str(y_pred_uncertain)
    ]
    wsheet.insert_row(row_data, len(data) + 1)
