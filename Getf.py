import torch

def getf(A):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    A = A.squeeze(0)
    n = A.size(0)
    mask = (A != 0) & (torch.eye(n, device=device) == 0)
    non_zero_elements = A[mask]
    result = torch.zeros((n, n - 1), dtype=A.dtype, device=device)  # 在 GPU 上创建结果矩阵
    for i in range(n):
        row_elements = non_zero_elements[
            (torch.div(torch.arange(non_zero_elements.size(0), device=device), (n - 1), rounding_mode='trunc')) == i]
        result[i, :row_elements.size(0)] = row_elements

    return result.unsqueeze(0)