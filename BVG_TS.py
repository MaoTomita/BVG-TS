import torch

class BVG_TS:
    """
    Bias-Variance Guided Tensor Selection (BVG_TS) class for fine-tuning neural networks.

    This class dynamically adjusts the learning rate of layers based on their bias-variance ratio (BVR),
    enabling efficient fine-tuning by prioritizing updates to layers with high BVR.
    """
    def __init__(self, optimizer, alpha=0.9):
        """
        Initialize the BVG_TS class.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose parameters will be adjusted.
            alpha (float): Exponential moving average coefficient for bias and variance updates.
        """
        self.optimizer = optimizer
        self.n_layers = len(optimizer.param_groups)  # Number of parameter groups (layers)
        self.moving_ave_dp = [[None for _ in param_group['params']] for param_group in optimizer.param_groups]
        self.moving_ave_dp2 = [[None for _ in param_group['params']] for param_group in optimizer.param_groups]
        
        self.alpha = alpha
        self.total_iter = 0  # Total number of iterations

    def set_lr(self, global_lr, update_ratio=0.5):
        """
        Adjust the learning rate for each tensor based on BVR.
        
        Args:
            global_lr (float): Base learning rate to be applied to selected tensors.
            update_ratio (float): Ratio of tensors to update based on the highest BVR (default is 0.5).
        
        Returns:
            tuple: Bias-variance ratios (torch.Tensor) and indices of updated tensors (torch.Tensor).
        """
        bvr = self._cal_bvr()
        # Calculate the number of tensors to update
        n_update = max(1, int(self.n_layers * update_ratio))
        
        # Identify tensors with the highest BVR
        top_tensor_idx = torch.argsort(bvr, descending=True)
        update_tensor_idx = top_tensor_idx[:n_update]
        
        # Set learning rate for each tensor
        for i in range(self.n_layers):
            if i in update_tensor_idx:
                self.optimizer.param_groups[i]['lr'] = global_lr
            else:
                self.optimizer.param_groups[i]['lr'] = 0
        
        return bvr, update_tensor_idx

    def _cal_bvr(self):
        """
        Calculate the bias-variance ratio (BVR) for each layer.

        Returns:
            torch.Tensor: Bias-variance ratios for all layers.
        """
        self.total_iter += 1
        dp_bias = torch.zeros(self.n_layers, device=self.optimizer.param_groups[0]['params'][0].device)
        dp_var = torch.zeros(self.n_layers, device=self.optimizer.param_groups[0]['params'][0].device)

        for g_idx, param_group in enumerate(self.optimizer.param_groups):
            for p_idx, param in enumerate(param_group['params']):
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                # Update moving averages
                if self.moving_ave_dp[g_idx][p_idx] is None:
                    self.moving_ave_dp[g_idx][p_idx] = (1.0 - self.alpha) * grad
                    self.moving_ave_dp2[g_idx][p_idx] = (1.0 - self.alpha) * grad ** 2
                else:
                    self.moving_ave_dp[g_idx][p_idx] *= self.alpha
                    self.moving_ave_dp[g_idx][p_idx] += (1.0 - self.alpha) * grad
                    self.moving_ave_dp2[g_idx][p_idx] *= self.alpha
                    self.moving_ave_dp2[g_idx][p_idx] += (1.0 - self.alpha) * grad ** 2
                
                # Bias and variance correction
                corrected_mean = self.moving_ave_dp[g_idx][p_idx] / (1 - self.alpha ** self.total_iter)
                corrected_mean_sq = self.moving_ave_dp2[g_idx][p_idx] / (1 - self.alpha ** self.total_iter)
                
                dp_bias[g_idx] += torch.sum(corrected_mean ** 2)
                dp_var[g_idx] += torch.sum(corrected_mean_sq - corrected_mean ** 2)

        dp_var = torch.clamp(dp_var, min=1e-10)
        bvr = dp_bias / dp_var

        return bvr
