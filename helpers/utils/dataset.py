import numpy as np
import pandas as pd
import streamlit as st
from helpers.utils.fuzzy import fuzzy_kmeans_fit_predict
from helpers.utils.projections import projections_dict
from helpers.drive import api


@st.cache(allow_output_mutation=True)
def generate_incomplete_dataset(seed, dataset_path, incomplete_column, na_frac) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    na_indexes = df.sample(frac=na_frac, random_state=seed).index
    df.loc[na_indexes, incomplete_column] = np.NaN
    return df


def get_df_incomplete(projection_key, X, dataset_settings, na_fraction_selectbox):
    df = generate_incomplete_dataset(st.session_state['dataset_generation_seed'],
                                     dataset_settings['path'],
                                     dataset_settings['incomplete_column'],
                                     na_fraction_selectbox)
    X_projected = projections_dict[projection_key](X)
    df['x'] = [el[0] for el in X_projected]
    df['y'] = [el[1] for el in X_projected]
    return df

@st.cache
# TODO: adapt for other clustering method than k-means
def find_uncertain_y_indexes(X, n_clusters, fuzzy_certainty_thres=0.5):
    fuzzy_labels = fuzzy_kmeans_fit_predict(X, n_clusters)
    max_labels = fuzzy_labels.max(axis=1)
    filter_indexes = max_labels < fuzzy_certainty_thres
    indexes = [ind for ind, element in enumerate(filter_indexes) if element == True]
    return indexes


def save_results(y_pred_uncertain, seed, dataset_name, na_fraction):
    api.save_results(
        st.session_state["gsheet"],
        y_pred_uncertain,
        st.session_state['id'],
        seed,
        dataset_name,
        na_fraction
    )