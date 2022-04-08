import pandas as pd
import streamlit as st
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helpers.streamlit.callbacks import update_point_color
from helpers.streamlit.utils import init_session_state, read_settings_from_file, validate, init_new_annotation_task
from helpers.utils.dataset import add_projection_dimensions_to_df_incomplete, save_results, generate_incomplete_dataset
from helpers.utils.imputation import impute_global_mean, impute_cluster_mean, impute_knn_mean, impute_cluster_knn_mean
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
        na_fraction_selectbox = st.selectbox('Select na_fraction', [0.1, 0.2, 0.3], key="na_fraction_selectbox")

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
            n_of_points_to_annotate = len(st.session_state['border_points'])
            if n_of_points_to_annotate > 0:

                current_null_index = st.session_state['border_points'][0]

                points_to_show_mask = df_incomplete[incomplete_column].notnull()
                points_to_show_mask[current_null_index] = True
                df_to_show = df_incomplete[points_to_show_mask]

                st.session_state['current_null_index'] = current_null_index
                plot_scatter(df_to_show, incomplete_column, reference_columns, current_null_index)

                value = st.slider("", float(df_to_show[incomplete_column].min()), float(df_to_show[incomplete_column].max()),
                                  step=dataset_settings['precision'], key='my_slider',  on_change=update_point_color, args=[df_incomplete, incomplete_column])

                f"Records to be labeled: {len(st.session_state['border_points'])}"
                st.progress(
                    st.session_state['annotated_points'] / (
                            st.session_state['annotated_points'] + n_of_points_to_annotate))

                if st.button('Submit'):
                    st.session_state['imputed_values'][current_null_index] = value
                    st.session_state['border_points'].pop(0)
                    st.session_state['annotated_points'] += 1
                    st.experimental_rerun()

                if st.button('Cancel'):
                    st.session_state['started'] = False
                    st.experimental_rerun()

            else:
                # Iteration finished
                na_indexes = list(st.session_state['imputed_values'].keys())
                global_means = impute_global_mean(df_incomplete, incomplete_column, na_indexes)
                cluster_means = impute_cluster_mean(df_incomplete, incomplete_column, na_indexes)
                knn = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
                cluster_knn = impute_cluster_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)

                real_values = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()

                rows = ["Annotator", "Mean", "Cluster mean", "knn", "cluster knn"]
                imputations = [st.session_state['imputed_values'], global_means, cluster_means, knn, cluster_knn]
                MAE = [mean_absolute_error(list(imputation.values()), real_values) for imputation in imputations]
                RMSE = [mean_squared_error(list(imputation.values()), real_values, squared=False) for imputation in imputations]
                results = pd.DataFrame(MAE, columns=["MAE"], index=rows)
                results["RMSE"] = RMSE
                st.table(results)

                if st.button("OK, SAVE RESULTS"):
                    st.session_state['started'] = False
                    save_results(st.session_state['imputed_values'],
                                 st.session_state['dataset_generation_seed'],
                                 dataset_name_selectbox,
                                 na_fraction_selectbox)

                    st.success('Your answers have been saved')
                    st.balloons()
                    time.sleep(1)
                    st.experimental_rerun()

        except ValueError:
            st.error("Click START after changing the dataset to restart the annotation process")
