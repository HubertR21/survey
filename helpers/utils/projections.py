from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px

@st.cache
def perform_pca_projection(X):
    tsne = PCA(n_components=2, random_state=0)
    X_projected = tsne.fit_transform(X)
    return X_projected

@st.cache
def perform_umap_projection(X):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    X_projected = umap_2d.fit_transform(X)
    return X_projected

@st.cache
def perform_tsne_projection(X):
    tsne = TSNE(n_components=2, random_state=0)
    X_projected = tsne.fit_transform(X)
    return X_projected


projections_dict = {
    "pca": perform_pca_projection,
    "umap": perform_umap_projection,
    "tsne": perform_tsne_projection,
}


def plot_scatter(df, incomplete_column, current_null_index):
    fig = px.scatter(df.sort_values(by=['y_pred']), x="x", y="y",
                     color=incomplete_column,
                     symbol="y_pred",
                     #hover_name=target_column,
                     width=1000, height=800)
    fig.layout.legend.y = 1
    fig.layout.legend.x = 1.2
    uncertain_x, uncertain_y = df[['x', 'y']].loc[current_null_index]
    add_annotation(fig, uncertain_x, uncertain_y)
    st.plotly_chart(fig)


def add_annotation(fig, x, y):
    fig.add_annotation(
        x=x,
        y=y,
        yshift=10,
        xref="x",
        yref="y",
        text="x",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=3,
        bgcolor="#636363",
        opacity=0.5
    )