# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:02:23 2025

@author: usouu
"""

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %% Visualization
def draw_heatmap_1d(data, yticklabels=None, xticklabels=None, figsize=(2, 10)):
    """
    Plots a heatmap for an Nx1 array (vertical orientation).

    Parameters:
        data (numpy.ndarray): Nx1 array for visualization.
        yticklabels (list, optional): Labels for the y-axis. If None, indices will be used.
    """
    if yticklabels is None:
        yticklabels = list(range(data.shape[0]))  # Automatically generate indices as labels
    if xticklabels is None:
        xticklabels = list(range(data.shape[1]))  # Automatically generate indices as labels
    
    if len(data.shape) == 1:
        data = np.reshape(data, (-1, 1))
    
    data = np.array(data, dtype=float)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        data, 
        cmap='Blues',
        cbar=False,
        annot=False,
        linewidths=0.5, 
        yticklabels=yticklabels,
        xticklabels=xticklabels
    )
    # plt.title("Vertical Heatmap of Nx1 Array")
    plt.show()

def draw_joint_heatmap_1d(data_dict):
    heatmap_labels = []
    heatmap_data = []

    for label, data in data_dict.items():
        heatmap_labels.append(label)
        heatmap_data.append(data)

    heatmap_data = np.vstack(heatmap_data) 
    heatmap_labels = np.array(heatmap_labels)
        
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=heatmap_labels, linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of cw_target and All cw_fitting Vectors")
    plt.xlabel("Channel Index")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def draw_projection_(sample_projection, title=None,
                    xticklabels=None, yticklabels=None,
                    show_colorbar=True, max_labels=20):
    """
    Visualizes data projections (common for both datasets).

    Parameters:
        sample_projection (np.ndarray): 2D or 3D matrix to visualize.
        title (str): Optional plot title.
        xticklabels (list): Optional list of x-axis labels.
        yticklabels (list): Optional list of y-axis labels.
        show_colorbar (bool): Whether to display the color bar.
        max_labels (int): Maximum number of labels allowed before auto-hiding.
    """
    if title is None:
        title = "2D Matrix Visualization"

    def apply_axis_labels(ax, xticks, yticks):
        # Auto-hide if too many labels
        if xticks is not None and len(xticks) <= max_labels:
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, rotation=90)
        else:
            ax.set_xticks([])

        if yticks is not None and len(yticks) <= max_labels:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks)
        else:
            ax.set_yticks([])

    def draw_single(matrix, plot_title):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='viridis')

        if show_colorbar:
            plt.colorbar(im, ax=ax)

        ax.set_title(plot_title)
        apply_axis_labels(ax, xticklabels, yticklabels)
        plt.tight_layout()
        plt.show()

    if sample_projection.ndim == 2:
        draw_single(sample_projection, title)

    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            draw_single(sample_projection[i], f"Channel {i + 1} Visualization")

    else:
        raise ValueError(
            f"The dimension of sample matrix for drawing is wrong, shape of sample: {sample_projection.shape}"
        )

def draw_projection(sample_projection, title=None,
                    xticklabels=None, yticklabels=None,
                    show_colorbar=True, max_labels=20,
                    title_position="upper"):
    """
    Visualizes 2D or 3D data projections.

    Parameters:
        sample_projection (np.ndarray): 2D matrix, or 3D array where each slice
            along axis 0 is visualized separately.
        title (str): Optional plot title for 2D input.
        xticklabels (list): Optional x-axis labels.
        yticklabels (list): Optional y-axis labels.
        show_colorbar (bool): Whether to display the color bar.
        max_labels (int): Maximum number of displayed labels before sparsifying
            with omission markers.
        title_position (str): Position of the title, either "upper" or "lower".
    """
    if title is None:
        title = "2D Matrix Visualization"

    if title_position not in ["upper", "lower"]:
        raise ValueError("title_position must be either 'upper' or 'lower'")

    def sparsify_labels_with_ellipsis(labels, max_labels):
        """
        Return sparse tick positions and labels with '…' inserted to indicate
        omitted regions.
        """
        n = len(labels)

        if n == 0:
            return [], []

        if n <= max_labels:
            return list(range(n)), list(labels)

        # Keep evenly spaced labels, always including first and last
        core_count = max(2, max_labels)
        idx = np.linspace(0, n - 1, num=core_count, dtype=int)
        idx = np.unique(idx).tolist()

        tick_positions = []
        tick_labels = []

        prev = None
        for current in idx:
            if prev is not None and current - prev > 1:
                # Put omission marker roughly in the skipped region
                ellipsis_pos = (prev + current) / 2
                tick_positions.append(ellipsis_pos)
                tick_labels.append("…")

            tick_positions.append(current)
            tick_labels.append(labels[current])
            prev = current

        return tick_positions, tick_labels

    def apply_axis_labels(ax, xticks, yticks):
        if xticks is not None:
            x_pos, x_lab = sparsify_labels_with_ellipsis(xticks, max_labels)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_lab, rotation=90)

        if yticks is not None:
            y_pos, y_lab = sparsify_labels_with_ellipsis(yticks, max_labels)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_lab)

    def apply_title(ax, plot_title):
        if title_position == "upper":
            ax.set_title(plot_title, pad=10)
        elif title_position == "lower":
            ax.set_xlabel(plot_title, labelpad=20)

    def plot_single(matrix, plot_title):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap="viridis")

        if show_colorbar:
            plt.colorbar(im, ax=ax)

        apply_title(ax, plot_title)
        apply_axis_labels(ax, xticklabels, yticklabels)
        plt.tight_layout()
        plt.show()

    if sample_projection.ndim == 2:
        plot_single(sample_projection, title)

    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            plot_single(sample_projection[i], f"Channel {i + 1} Visualization")

    else:
        raise ValueError(
            f"The dimension of sample matrix for drawing is wrong, "
            f"shape of sample: {sample_projection.shape}"
        )
        
# %% Example Usage
# if __name__ == '__main__':
