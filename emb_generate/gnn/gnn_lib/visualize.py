import networkx as nx
import plotly.graph_objects as go
import numpy as np

def plot_cooccurrence_graph(id2treatment, treatment2id, cooccurrence_counter):
    G = nx.Graph()
    for idx, treatment in id2treatment.items():
        G.add_node(idx, label=treatment)

    for (src, dst), weight in cooccurrence_counter.items():
        G.add_edge(treatment2id[src], treatment2id[dst], weight=weight)

    pos = nx.spring_layout(G, seed=42)
    x_nodes = [pos[i][0] for i in G.nodes()]
    y_nodes = [pos[i][1] for i in G.nodes()]
    labels = [G.nodes[i]['label'] for i in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    norm_weights = [(w - min_weight) / (max_weight - min_weight + 1e-6) for w in edge_weights]
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        opacity=0.5, mode='lines', hoverinfo='none'
    )
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        text=labels,
        textposition='top center',
        hoverinfo='text',
        marker=dict(color='skyblue', size=12, line=dict(width=2, color='black'))
    )
    layout = go.Layout(
        title=dict(text='Treatment Co-occurrence Graph', font=dict(size=18)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.write_html("treatment_cooccurrence_graph.html")
    print("Saved to treatment_cooccurrence_graph.html")
    return fig

def plot_treatment_embeddings_tsne(final_treatment_embeddings, id2treatment, out_html="treatment_embeddings_visualization.html"):
    from sklearn.manifold import TSNE
    import plotly.express as px

    # Move embeddings to CPU and detach
    embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(
        final_treatment_embeddings.detach().cpu().numpy()
    )

    num_treatments = embeddings_2d.shape[0]
    hover_labels = [id2treatment[idx] for idx in range(num_treatments)]

    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hover_name=hover_labels,
        title="Treatment Embedding Visualization (t-SNE)",
        width=800,
        height=600,
    )

    fig.update_traces(marker=dict(size=8))
    fig.write_html(out_html)
    print(f"Saved t-SNE visualization to {out_html}")

    return fig