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
    uncertain_x, uncertain_y = df[['x', 'y']].loc[current_null_index]

    fig = px.scatter(df, x="x", y="y",
                     color=incomplete_column,
                     color_continuous_scale=px.colors.sequential.Inferno,
                     # symbol="y_pred",
                     # symbol_sequence=["circle", "square", "diamond", "cross", "triangle-up", "triangle-down",
                     #                  "pentagon", "circle-cross", "square-cross", "diamond-cross"],
                     symbol_map={"uncertain": "x"},
                     # text='y_pred',
                     hover_data={'x': False,
                                 'y': False,
                                 incomplete_column: ':.2f'
                                 },
                     range_x=[uncertain_x - df['x'].std(), uncertain_x + df['x'].std()],
                     range_y=[uncertain_y - df['y'].std(), uncertain_y + df['y'].std()],
                     # hover_name=target_column,
                     height=600,
                     width=800
                     )
    fig.update_traces(textposition='top center', textfont_size=8,
                      textfont_color="#636363", marker_size=12
                      )

    fig.layout.showlegend = False
    # fig.update_layout(hovermode="incomplete_column")

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
