import numpy as np
import matplotlib.pyplot as plt


def visualize_weights(weight_matrix, vin, vout, labels=None):
    labels = list(range(weight_matrix.shape[0])) if labels is None else labels
    vector = np.array(range(weight_matrix.shape[0]))

    # Ensure vector is a row vector
    vector = vector.reshape(1, -1)
    w, h, ppi = 600, 300, 100
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(w / float(ppi), h / float(ppi)),
        dpi=ppi,
        gridspec_kw={"height_ratios": [0.25, 0.5, 0.25]},
    )
    # Decrease vertical padding between subplots
    fig.subplots_adjust(hspace=0.55)

    ax1.bar(labels, vin)

    target_vector = vector.copy()
    # Plot source vector; zorder to be on top
    ax2.scatter(vector, np.zeros_like(vector), c="k", zorder=10)
    # Plot target vector
    ax2.scatter(target_vector, np.ones_like(vector), c="k", label="Target", zorder=10)

    for ax in [ax1, ax3]:
        ax.set_ylim([-1, 1])

    ax2.set_yticks([])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)

    ax3.set_xticks([])

    for ax in [ax1, ax2, ax3]:
        # Horizontal alignment
        ax.set_xlim([-0.2, len(vin) - 0.8])

        # No ticks for this visualization
        ax.set_xticks([])

    for i, row_weights in enumerate(weight_matrix):
        # Create lines with intensities based on weights
        plt.sca(ax2)

        # Draw lines with weight-based intensity
        for j, weight in enumerate(row_weights):
            plt.plot(
                [i, j],
                [0, 1],
                color="tab:blue",
                alpha=abs(weight),  # Use absolute weight for line intensity
                linewidth=abs(weight) * 5,
            )  # Line thickness proportional to weight

    ax3.bar(labels, vout)

    ax1_ = ax1.twinx()
    ax1_.set_yticks([])
    ax1_.set_ylabel("In", rotation=0, labelpad=15)

    ax3_ = ax3.twinx()
    ax3_.set_yticks([])
    ax3_.set_ylabel("Out", rotation=0, labelpad=15)

    return fig


if __name__ == "__main__":
    # Example usage
    weight_matrix = np.array([[0.5, 0.3, 0.7], [0.2, 0.2, 0.2], [0.9, 0.1, 0.1]])

    visualize_weights(
        weight_matrix,
        np.random.rand(3),
        np.random.rand(3),
        labels=["Feature 1", "Feature 2", "Feature 3"],
    )
    plt.savefig("weights.png")
