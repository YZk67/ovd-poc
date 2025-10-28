import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
from matplotlib.colors import ListedColormap

def visualize_rpsa(image, pi, s, r, token_coords,
                   class_names=None,
                   proto_texts=None,
                   gt_classes=None,
                   topk_classes=5,
                   save_path=None):
    """
    Visualize RPSA interpretability: π heatmap, s heatmap, and spatial cluster map.
    Args:
        image: np.ndarray [H, W, 3] 原始图像
        pi: torch.Tensor [K, C] cluster→class 权重
        s: torch.Tensor [K, C, Kp] cluster→prototype 相似度
        r: torch.Tensor [N, K] token→cluster assignment
        token_coords: np.ndarray [N, 2] 每个 token 在图上的中心坐标 (x, y)
        class_names: list[str], 类别名称
        proto_texts: list[list[str]], 每类 prototype 对应的 prompt
        gt_classes: list[int], 当前图像中出现的类
        topk_classes: 只画相似度最高的前 k 个类
        save_path: 若非 None，保存图片
    """
    device = 'cpu'
    if torch.is_tensor(pi): pi = pi.detach().to(device).numpy()
    if torch.is_tensor(s):  s  = s.detach().to(device).numpy()
    if torch.is_tensor(r):  r  = r.detach().to(device).numpy()

    K, C = pi.shape
    Kp = s.shape[-1]

    # =======================
    # Figure Layout
    # =======================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
    ax_pi = fig.add_subplot(gs[0, 0])
    ax_s  = fig.add_subplot(gs[0, 1])
    ax_sp = fig.add_subplot(gs[0, 2])
    ax_img = fig.add_subplot(gs[1, :])

    # =======================
    # 1️⃣ π heatmap (Cluster–Class)
    # =======================
    top_classes = np.argsort(pi.mean(0))[::-1][:topk_classes]
    pi_show = pi[:, top_classes]
    sns.heatmap(pi_show, cmap="YlGnBu", ax=ax_pi, cbar=True)
    ax_pi.set_title("Cluster–Class Weights (π)")
    ax_pi.set_xlabel("Top Classes")
    ax_pi.set_ylabel("Clusters")
    if class_names:
        ax_pi.set_xticklabels([class_names[c] for c in top_classes], rotation=45, ha='right')

    # =======================
    # 2️⃣ s heatmap (Cluster–Prototype)
    # =======================
    # 取 topk 类的 prototype 相似度
    s_show = s[:, top_classes, :].reshape(K, -1)
    sns.heatmap(s_show, cmap="RdYlBu_r", ax=ax_s, cbar=True)
    ax_s.set_title("Cluster–Prototype Similarity (s)")
    ax_s.set_xlabel("Prototypes")
    ax_s.set_ylabel("Clusters")
    if proto_texts and class_names:
        xticks = []
        for c in top_classes:
            for j in range(len(proto_texts[c])):
                xticks.append(f"{class_names[c]}#{j+1}")
        ax_s.set_xticklabels(xticks, rotation=45, ha='right')

    # =======================
    # 3️⃣ π purity / entropy plot
    # =======================
    entropy = -(pi * np.log(pi + 1e-8)).sum(1) / np.log(C)
    purity  = 1 - entropy
    ax_sp.bar(np.arange(K), purity, color='coral')
    ax_sp.set_title("Cluster Purity (1−Entropy)")
    ax_sp.set_xlabel("Cluster")
    ax_sp.set_ylabel("Purity")

    # =======================
    # 4️⃣ Spatial map overlay
    # =======================
    overlay = image.copy().astype(np.float32) / 255
    cluster_id = np.argmax(r, axis=1)
    cmap = ListedColormap(sns.color_palette("husl", K))
    for k in range(K):
        pts = token_coords[cluster_id == k]
        if len(pts) == 0:
            continue
        color = np.array(cmap(k)[:3])
        for (x, y) in pts:
            cv2.circle(overlay, (int(x), int(y)), 2, color.tolist(), -1)
    blended = (0.6 * overlay + 0.4 * (image.astype(np.float32)/255)).clip(0,1)
    ax_img.imshow(blended)
    ax_img.set_title("Spatial Cluster Assignment (r)")
    ax_img.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[RPSA-Vis] saved → {save_path}")
    plt.show()


# from rpsa_visualizer import visualize_rpsa

# visualize_rpsa(
#     image=img,                     # H×W×3 numpy array (原图)
#     pi=pi_tensor,                  # [K, C]
#     s=s_tensor,                    # [K, C, Kp]
#     r=r_tensor,                    # [N, K] (token→cluster)
#     token_coords=token_xy,         # [N, 2] 对应 encoder token 在图上的坐标
#     class_names=lvis_classes,      # 类别名列表
#     proto_texts=proto_prompts,     # 每类的 prototype 文本（list[list[str]]）
#     gt_classes=gt_cls_ids,         # 当前图像的 GT 类 id 列表
#     save_path="vis_rpsa.png"       # 输出路径
# )
