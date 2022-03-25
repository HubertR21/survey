import streamlit as st
import time

from helpers.streamlit.callbacks import update_point_color
from helpers.streamlit.utils import init_session_state, read_settings_from_file, validate, init_new_annotation_task
from helpers.utils.dataset import add_projection_dimensions_to_df_incomplete
from helpers.utils.projections import projections_dict, plot_scatter



def get_y_pred_uncertain_str():
    return [str(y_pred) if y_pred is not None else None for y_pred in st.session_state['y_pred_uncertain']]


st.set_page_config(layout="wide")
st.sidebar.markdown('## Authorization')
user_id = st.sidebar.text_input('Your ID')
user_password = st.sidebar.text_input('Password', type="password")


if not validate(user_id) or not user_password == st.secrets['password']:
    '# Please authorize'
    'Please provide your ID and password in the sidebar inputs.'
else:
    datasets_settings = read_settings_from_file()
    init_session_state(datasets_settings)

    with st.sidebar:
        st.markdown('## Settings')
        dataset_name_selectbox = st.selectbox('Select dataset', [dataset['name'] for dataset in datasets_settings])
        na_fraction_selectbox = st.selectbox('Select na_fraction', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], key="na_fraction_selectbox")

    dataset_settings = next(dataset for dataset in datasets_settings if dataset['name'] == dataset_name_selectbox)
    incomplete_column = dataset_settings['incomplete_column']
    reference_columns = dataset_settings['reference_columns']

    if st.sidebar.button('START'):
        st.session_state.df_incomplete = init_new_annotation_task(dataset_settings)

    if st.session_state['started']:
        with st.sidebar:
            projection_key = st.selectbox('Set projection', projections_dict.keys())
        try:
            df_incomplete = add_projection_dimensions_to_df_incomplete(str(projection_key),
                                                                       st.session_state.X,
                                                                       st.session_state.df_incomplete)

            df_incomplete['y_pred'] = get_y_pred_uncertain_str()

            df_null_indexes = df_incomplete[df_incomplete['y_pred'].isnull()].index.tolist()
            n_of_points_to_annotate = len(df_null_indexes)
            if n_of_points_to_annotate > 0:

                current_null_index = df_null_indexes[0]

                cluster_labels = [el for el in df_incomplete['y_pred'].unique() if el is not None]

                df_incomplete.at[current_null_index, 'y_pred'] = 'uncertain'
                df_to_show = df_incomplete[df_incomplete['y_pred'].notnull()]
                st.session_state['current_null_index'] = current_null_index
                plot_scatter(df_to_show, incomplete_column, reference_columns, current_null_index)

                value = st.slider("", float(df_to_show[incomplete_column].min()), float(df_to_show[incomplete_column].max()),
                                  step=dataset_settings['precision'], key='my_slider',  on_change=update_point_color, args=[df_incomplete, incomplete_column])

                f"Records to be labeled: {len(df_null_indexes)}"
                st.progress(
                    st.session_state['annotated_points'] / (
                            st.session_state['annotated_points'] + n_of_points_to_annotate))

                if st.button('Submit'):
                    real_value=st.session_state[f'ds_{dataset_settings["name"]}'].at[current_null_index, incomplete_column]
                    st.session_state['annotator_error'] += abs(real_value - value)
                    st.session_state['mean_error'] += abs(real_value - df_incomplete[incomplete_column].mean())

                    st.session_state['y_pred_uncertain'][current_null_index] = int(0) # fix this hack
                    st.session_state['annotated_points'] += 1
                    st.experimental_rerun()
                if st.button('Cancel'):
                    st.session_state['annotated_points'] = 0
                    st.session_state['started'] = False
                    st.experimental_rerun()

            else:
                # Iteration finished
                # save_results(st.session_state['y_pred_uncertain'],
                #              st.session_state['dataset_generation_seed'],
                #              dataset_name_selectbox,
                #              na_fraction_selectbox)
                print(f"Annotator error: {st.session_state['annotator_error']}")
                print(f"Mean error: {st.session_state['mean_error']}")
                f"Annotator error: {st.session_state['annotator_error']}"
                f"Mean error: {st.session_state['mean_error']}"
                if st.button("OK"):
                    st.session_state['annotated_points'] = 0
                    st.session_state['started'] = False
                    st.session_state['dataset_generation_seed'] = int(time.time())
                    # st.success('Your answers have been saved')
                    st.balloons()
                    time.sleep(1)

                    st.experimental_rerun()
        except ValueError:
            st.error("Click START after changing the dataset to restart the annotation process")
