import torch
import torch.nn.functional as F
import kmeans_tools

def get_dist_argmin_half_batched(D):
    if D == 4:
        return kmeans_tools.dist_argmin_half_batched_d4
    elif D == 8:
        return kmeans_tools.dist_argmin_half_batched_d8
    elif D == 9:
        return kmeans_tools.dist_argmin_half_batched_d9
    elif D == 10:
        return kmeans_tools.dist_argmin_half_batched_d10
    else:
        raise ValueError(f"Unsupported dimension: {D}")


def kmeans_plusplus_batch(X, k):
    """
    Batch kmeans++ initialization with incremental distance updates.
    X: Tensor of shape (B, N, D)
    Returns: centroids of shape (B, k, D)
    """
    B, N, D = X.shape
    device = X.device

    centroids = torch.empty((B, k, D), device=device)

    # First centroid: randomly chosen for each batch.
    random_idx = torch.randint(0, N, (B,), device=device)
    centroids[:, 0] = X[torch.arange(B, device=device), random_idx]

    # Compute initial distances from the first centroid.
    min_dists = torch.cdist(X, centroids[:, 0:1]).squeeze(2)  # (B, N)

    for i in range(1, k):
        # Compute probability distribution proportional to the squared distance.
        probs = min_dists ** 2
        sum_probs = probs.sum(dim=1, keepdim=True)
        probs = probs / (sum_probs + 1e-8)
        
        # Sample the next centroid index for each batch.
        new_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        centroids[:, i] = X[torch.arange(B, device=device), new_idx]
        
        # Update the minimum distances using the new centroid.
        new_dists = torch.cdist(X, centroids[:, i:i+1]).squeeze(2)
        min_dists = torch.minimum(min_dists, new_dists)
        
    return centroids


def weighted_kmeans_batch(X, weights, k, num_iters=10):
    """
    Batched weighted k-means clustering.
    X: Tensor of shape (B, N, D)
    weights: Tensor of shape (B, N)
    k: number of clusters
    num_iters: maximum iterations
    Returns: centroids of shape (B, k, D) and labels of shape (B, N)
    """
    B, N, D = X.shape
    device = X.device

    # KMeans++ initialization
    centroids = kmeans_plusplus_batch(X, k)

    # Select the appropriate half-batched distance function.
    dist_argmin_half_batched = get_dist_argmin_half_batched(D)

    for it in range(num_iters):
        # Cluster assignment step using the custom distance kernel.
        labels = dist_argmin_half_batched(X.half(), centroids.half())  # (B, N)
        # Compute weighted sums and counts for centroids using scatter_add.
        weighted_sum = torch.zeros(B, k, D, device=device)
        weighted_count = torch.zeros(B, k, device=device)
        
        # Scatter the weighted sums: expand labels to (B, N, 1) to match X’s dimensions.
        weighted_sum.scatter_add_(1, labels.unsqueeze(-1).expand(B, N, D).long(), weights.unsqueeze(-1) * X)
        weighted_count.scatter_add_(1, labels.long(), weights)
        
        # Update centroids: compute the weighted average.
        centroids_new = weighted_sum / (weighted_count.unsqueeze(-1) + 1e-8)

        zero_mask = (centroids_new == 0).all(dim=-1)  # shape: (B, k)
        rand_indices = torch.randint(0, N, size=(B, k), device=X.device)  # shape: (B, k)
        reinit_vectors = torch.gather(
            X, 1, rand_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # shape: (B, k, D)
        centroids_new = torch.where(
            zero_mask.unsqueeze(-1),  # shape: (B, k, 1)
            reinit_vectors,
            centroids_new
        )

        # Convergence check: if the mean absolute change is small enough, break early.
        if torch.abs(centroids_new - centroids).mean() < 1e-5:
            print(f"Converged at iteration {it}/{num_iters}")
            return centroids_new, labels

        centroids = centroids_new

    return centroids, labels
