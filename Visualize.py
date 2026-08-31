
import matplotlib.pyplot as plt
import numpy as np
import torch

COLOR_PALETTE = np.array([
                        [0, 0, 0],  # 0: 背景
                        [0, 255, 0],  # 1: 健康草地
                        [255, 0, 0],  # 2: 受压草地
                        [0, 0, 255],  # 3: 人工草地
                        [255, 255, 0],  # 4: 树木
                        [128, 0, 128],  # 5: 土壤
                        [0, 255, 255],  # 6: 水
                        [255, 128, 0],  # 7: 住宅区
                        [128, 255, 0],  # 8: 商业区
                        [255, 0, 255],  # 9: 道路
                        [0, 128, 255],  # 10: 高速公路
                        [128, 128, 128],  # 11: 铁路
                        [64, 128, 0],  # 12: 停车场1
                        [0, 64, 128],  # 13: 停车场2
                        [128, 0, 64],  # 14: 网球场
                        [192, 192, 64]  # 15: 跑道
                    ], dtype=np.uint8)
def visualize_epoch(epoch, pred_tensor, combined_labels):
    # 生成伪标签
    with torch.no_grad():
        pseudo_labels = torch.max(torch.softmax(pred_tensor, dim=0), dim=0)[1]
        mask = (combined_labels > 0).float()
        pseudo_labels = (pseudo_labels.cpu() * mask.cpu()).numpy().astype(np.uint8)
    colored_labels = COLOR_PALETTE[pseudo_labels]
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(colored_labels)
    plt.axis('off')
    filename = f'immhh/output_epoch_{epoch:03d}.png'  # 三位数编号
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    # 清理资源
    plt.close(fig)
    del fig