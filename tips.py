import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap

# === Data Load ===
data = pd.read_csv("tips.csv")

# === Program-8: Hierarchical Clustering ===


def plot_dendrogram():
    X_cluster = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    Z = linkage(X_cluster, method='ward')
    plt.figure()
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Point Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

# === Program-9: Visualization Techniques ===


def plot_bar_chart():
    tips_by_day = data.groupby('day')['tip'].sum()
    tips_by_day.plot(kind='bar', title='Total Tips by Day', color='skyblue')
    plt.xlabel('Day')
    plt.ylabel('Total Tip')
    plt.tight_layout()
    plt.show()


def plot_line_chart():
    sns.lineplot(x='day', y='tip', data=data, marker='o')
    plt.title('Tip Trend by Day')
    plt.tight_layout()
    plt.show()


def plot_scatter_bokeh():
    source = ColumnDataSource(data)
    graph = figure(title="Bokeh Scatter: Total Bill vs Tip",
                   width=700, height=400)
    graph.scatter(x='total_bill', y='tip', source=source,
                  color=factor_cmap(
                      'gender', palette=['blue', 'orange'], factors=sorted(data['gender'].unique())),
                  legend_field='gender')
    graph.xaxis.axis_label = "Total Bill"
    graph.yaxis.axis_label = "Tip"
    show(graph)


def plot_histogram():
    sns.histplot(x='total_bill', data=data, kde=True,
                 hue='gender', palette='muted')
    plt.title("Histogram of Total Bill by gender")
    plt.tight_layout()
    plt.show()


def plot_3d_surface():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X_grid**2 + Y_grid**2))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(
        X_grid, Y_grid, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    ax.set_title('3D Surface Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()


# === Execute All ===
plot_dendrogram()
plot_bar_chart()
plot_line_chart()
plot_scatter_bokeh()
plot_histogram()
plot_3d_surface()
