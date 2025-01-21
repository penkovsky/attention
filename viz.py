import numpy as np
import matplotlib.pyplot as plt


def visualize_weights(weight_matrix, v, labels=None):
    vector = np.array(range(weight_matrix.shape[0]))

    # Ensure vector is a row vector
    vector = vector.reshape(1, -1)
    w, h, ppi = 600, 280, 100
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(w/float(ppi), h/float(ppi)),
                                   dpi=ppi,
                                   gridspec_kw={"height_ratios": [0.25, 0.75]})

    ax1.bar(labels, v)

    target_vector = vector.copy()
    # Plot source vector; zorder to be on top
    plt.scatter(vector, np.zeros_like(vector), c="k", zorder=10)
    # Plot target vector
    plt.scatter(target_vector, np.ones_like(vector), c="k", label="Target", zorder=10)

    ax1.set_ylim([-1, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    for ax in [ax1, ax2]:
        # Horizontal alignment
        ax.set_xlim([-0.2, len(v) - 0.8])

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

    plt.tight_layout()

if __name__ == "__main__":
    # Example usage
    weight_matrix = np.array([[0.5, 0.3, 0.7], [0.2, 0.2, 0.2], [0.9, 0.1, 0.1]])

    visualize_weights(
        weight_matrix, np.random.rand(3), labels=["Feature 1", "Feature 2", "Feature 3"]
    )
    plt.savefig("weights.png")
