import pandas as pd
from sklearn.neighbors import NearestNeighbors


def weighted_average(distribution, weights):
    numerator = sum([distribution[i] * weights[i] for i in range(len(distribution))])
    denominator = sum(weights)

    return round(numerator / denominator, 2)


def impute_global_mean(df, incomplete_column, na_indexes):
    mean = df[incomplete_column].mean()
    return {idx: mean for idx in na_indexes}


def impute_knn_mean_helper(df, na_indexes, reference_columns, incomplete_column, n_neighbors=5,
                           weight_func=lambda x: 1):
    not_na_mask = [x not in na_indexes for x in df[incomplete_column].index]

    if len(na_indexes) > 0:
        # performing KNN for each NA record
        # neigh_distances, neigh_indexes are arays - their elements corresponds to NA records
        # each element of those arrays is also an array containing 'n_neighbors' nearest neighbors details (distances and indexes)
        # WARNING: neigh_indexes contains 'row numbers' relative to 'df_not_na' (which was fitted by 'neigh')
        # it means that neigh_indexes are numbers of rows in df_not_na, NOT indexes from original df
        # TODO: refactor this - use class not arrays in arrays to be more readable
        na_records = df[reference_columns].loc[na_indexes]
        df_not_na = df[not_na_mask][reference_columns]
        neigh = NearestNeighbors(n_neighbors=min(len(df_not_na), n_neighbors))
        neigh.fit(df_not_na)
        neigh_distances, neigh_indexes = neigh.kneighbors(na_records)

        # for ind, neigh_ind, neigh_dist in zip(na_indexes, neigh_indexes, neigh_distances):
        # getting df indexes based on neigh_indexes (see WARNING above) df_ind = df_not_na.index[neigh_ind]
        return {idx: weighted_average(list(df.loc[df_not_na.index[neigh_ind]][incomplete_column]),
                                      [weight_func(dist) for dist in neigh_dist])
                for idx, neigh_ind, neigh_dist in zip(na_indexes, neigh_indexes, neigh_distances)}


def impute_knn_mean(df, incomplete_column, na_indexes, reference_columns, n_neighbors=5, weight_func=lambda x: 1):
    return impute_knn_mean_helper(df, na_indexes, reference_columns, incomplete_column, n_neighbors, weight_func)


def impute_cluster_mean(df, incomplete_column, na_indexes):
    cluster_means = []
    number_of_clusters = df['y_pred'].max() + 1
    for label in range(number_of_clusters):
        curent_cluster_mask = df['y_pred'] == label
        cluster_means.append(df[curent_cluster_mask][incomplete_column].mean())

    return {idx: cluster_means[df['y_pred'][idx]] for idx in na_indexes}


def impute_cluster_knn_mean(df, incomplete_column, na_indexes, reference_columns, n_neighbors=5,
                            weight_func=lambda x: 1):
    values = {}
    number_of_clusters = df['y_pred'].max() + 1
    for label in range(number_of_clusters):
        current_cluster_mask = df['y_pred'] == label
        df_current_cluster = df[current_cluster_mask]

        res = impute_knn_mean_helper(df_current_cluster,
                                     list(set(na_indexes).intersection(df_current_cluster.index.tolist())),
                                     reference_columns, incomplete_column, n_neighbors, weight_func)

        if res:
            values.update(res)

    return values
