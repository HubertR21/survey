import streamlit as st
import json
import time

from helpers.drive.api import GSAuthentication
from helpers.utils.dataset import generate_incomplete_dataset, get_df_incomplete, save_results, find_uncertain_y_indexes
from helpers.utils.projections import projections_dict, plot_scatter
from sklearn.cluster import KMeans


def validate(user_id):
    validated = user_id.isdigit()
    if validated:
        st.session_state['id'] = user_id
    return validated


def calculate_y_pred_uncertain():
    X = df_incomplete[reference_columns].values

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    uncertain_y_indexes = find_uncertain_y_indexes(X, number_of_clusters)
    y_pred_uncertain = [int(el) if i not in uncertain_y_indexes else None for i, el in enumerate(y_pred)]
    return X, y_pred_uncertain


def update_session_state(X, y_pred_uncertain):
    st.session_state.X = X
    st.session_state['y_pred_uncertain'] = y_pred_uncertain


def get_y_pred_uncertain_str():
    return [str(y_pred) if y_pred is not None else None for y_pred in st.session_state['y_pred_uncertain']]


st.set_page_config(layout="wide")
st.sidebar.markdown('## Authorization')
user_id = st.sidebar.text_input('Your ID')
user_password = st.sidebar.text_input('Password', type="password")

st.session_state["gsheet"] = GSAuthentication()

if not validate(user_id) or not user_password == st.secrets['password']:
    '# Please authorize'
    'Please provide your ID and password in a sidebar inputs.'
else:
    with open("settings.json", "r") as f:
        settings = json.load(f)

    datasets_settings = settings['data']

    if 'started' not in st.session_state:
        st.session_state['started'] = False
    if 'annotated_points' not in st.session_state:
        st.session_state['annotated_points'] = 0

    with st.sidebar:
        st.markdown('## Settings')
        dataset_name_selectbox = st.selectbox('Select dataset', [dataset['name'] for dataset in datasets_settings])
        na_fraction_selectbox = st.selectbox('Select na_fraction', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    dataset_settings = next(dataset for dataset in datasets_settings if dataset['name'] == dataset_name_selectbox)

    number_of_clusters = dataset_settings['number_of_clusters']
    reference_columns = dataset_settings['reference_columns']
    incomplete_column = dataset_settings['incomplete_column']
    target_column = dataset_settings['target_column']

    if st.sidebar.button('START'):
        if 'dataset_generation_seed' not in st.session_state:
            st.session_state['dataset_generation_seed'] = int(time.time())

        df_incomplete = generate_incomplete_dataset(st.session_state['dataset_generation_seed'],
                                                    dataset_settings['path'],
                                                    incomplete_column,
                                                    na_fraction_selectbox)
        update_session_state(*calculate_y_pred_uncertain())
        st.session_state['started'] = True

    if st.session_state['started']:
        projection_key = st.selectbox('Set projection', projections_dict.keys())
        try:
            df_incomplete = get_df_incomplete(str(projection_key), st.session_state.X, dataset_settings,
                                              na_fraction_selectbox)

            df_incomplete['y_pred'] = get_y_pred_uncertain_str()

            df_null_indexes = df_incomplete[df_incomplete['y_pred'].isnull()].index.tolist()
            n_of_points_to_annotate = len(df_null_indexes)
            if n_of_points_to_annotate > 0:

                current_null_index = df_null_indexes[0]

                cluster_labels = [el for el in df_incomplete['y_pred'].unique() if el is not None]

                df_incomplete.at[current_null_index, 'y_pred'] = 'uncertain'
                df_to_show = df_incomplete[df_incomplete['y_pred'].notnull()]
                plot_scatter(df_to_show, incomplete_column, current_null_index)

                with st.sidebar:
                    f"Records to be labeled: {len(df_null_indexes)}"
                    st.progress(
                        st.session_state['annotated_points'] / (
                                st.session_state['annotated_points'] + n_of_points_to_annotate))

                    uncertain_label = st.radio('Select label for uncertain record', cluster_labels)
                    if st.button('Submit'):
                        st.session_state['y_pred_uncertain'][current_null_index] = int(uncertain_label)
                        st.session_state['annotated_points'] += 1
                        st.experimental_rerun()
                    if st.button('Cancel'):
                        st.session_state['annotated_points'] = 0
                        st.session_state['started'] = False
                        st.experimental_rerun()
            else:
                # Iteration finished
                save_results(st.session_state['y_pred_uncertain'],
                             st.session_state['dataset_generation_seed'],
                             dataset_name_selectbox,
                             na_fraction_selectbox)

                st.session_state['annotated_points'] = 0
                st.session_state['started'] = False
                st.session_state['dataset_generation_seed'] = int(time.time())

                st.success('Your answers have been saved')
                st.balloons()
                st.experimental_rerun()
        except ValueError:
            st.error("Click START after changing the dataset to restart the annotation process")
