import torch

def kmeans_plusplus_batch(X, k):
    B, N, D = X.shape
    device = X.device

    centroids = torch.zeros((B, k, D), device=device)

    # 첫 번째 중심: 무작위 선택
    idx = torch.randint(0, N, (B,), device=device)
    centroids[:, 0] = X[torch.arange(B), idx]

    for i in range(1, k):
        # 이전까지의 centroid들과 거리 계산
        prev = centroids[:, :i]                      # (B, i, D)
        dist = torch.cdist(X, prev)                  # (B, N, i)
        min_dist, _ = dist.min(dim=2)                # (B, N)
        probs = min_dist ** 2
        probs = probs / probs.sum(dim=1, keepdim=True)

        # 확률적으로 다음 centroid 선택
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        centroids[:, i] = X[torch.arange(B), next_idx]

    return centroids  # (B, k, D)


def weighted_kmeans_batch(X, weights, k, num_iters=10):
    B, N, D = X.shape
    device = X.device

    # KMeans++ 초기화
    centroids = kmeans_plusplus_batch(X, k)  # (B, k, D)

    for it in range(num_iters):
        # 거리 계산: (B, N, k)
        dists = torch.cdist(X, centroids)  # (B, N, k)
        labels = dists.argmin(dim=2)       # (B, N)

        centroids_new = torch.zeros_like(centroids)

        for i in range(k):
            mask = labels == i  # (B, N)
            w_mask = mask * weights  # (B, N)
            w_mask_unsq = w_mask.unsqueeze(2)  # (B, N, 1)

            numerator = (X * w_mask_unsq).sum(dim=1)  # (B, D)
            denominator = w_mask.sum(dim=1, keepdim=True) + 1e-8  # (B, 1)

            centroid_i = numerator / denominator  # (B, D)
            centroids_new[:, i] = centroid_i

        if (centroids_new - centroids).abs().mean() < 1e-5:
            print(f"{it}/{num_iters}")
            return centroids_new, labels

        centroids = centroids_new

    return centroids, labels  # (B, k, D), (B, N)
