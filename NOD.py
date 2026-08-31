
import torch

def nod(A):

    I = torch.eye(A.size(1), device=A.device)
    A_tilde = A + I
    degrees = torch.sum(A_tilde, dim=2)
    D_tilde = torch.diag_embed(degrees)
    D_tilde_inv_sqrt = torch.diag_embed(torch.pow(degrees, -0.5))
    A_norm = torch.bmm(torch.bmm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
    return A_norm