import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias, outlier_indices):
        super().__init__()
        
        # split weight into outlier / normal columns
        W_outlier = weight[:, outlier_indices]  # shape: [out_features, num_outliers]
        mask = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
        mask[outlier_indices] = False
        W_normal = weight[:, mask]  
        normal_indices = torch.arange(weight.shape[1], device=weight.device)[mask]
        # quantize W_normal to INT8 
        row_max = W_normal.abs().max(dim=-1).values  # shape: [out_features]
        scale = 127 / row_max  # shape: [out_features]

        W_normal_int8 = torch.round(W_normal * scale.unsqueeze(1)).to(torch.int8)

        # store everything as buffers 
        self.register_buffer('W_outlier', W_outlier)
        self.register_buffer('W_normal_int8', W_normal_int8)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        # also registering the outlier and normal indices because in forward, we will need to split the input as well so we use these indices
        self.register_buffer('outlier_indices', outlier_indices) 
        self.register_buffer('normal_indices', normal_indices)
        
    def forward(self, x):

        # split x into x_outlier / x_normal 
        x_outlier = x[..., self.outlier_indices]
        x_normal = x[..., self.normal_indices]

        # quantize x_normal
        row_max = x_normal.abs().max(dim=-1).values 
        scale = 127 / row_max 
        X_normal_int8 = torch.round(x_normal * scale.unsqueeze(-1)).to(torch.int8)
        
        # two matmuls: FP16 path + INT8 path (dequantized)
        y_outlier = x_outlier @ self.W_outlier.T
        y_int8 = torch.matmul(X_normal_int8.float(), self.W_normal_int8.float().T)
        y_normal = y_int8 / (scale.unsqueeze(-1) * self.scale) # dequantize
        # finally adding teh result and bias
        output = y_outlier + y_normal + self.bias

        return output 