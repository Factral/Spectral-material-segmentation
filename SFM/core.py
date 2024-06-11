import torch
import torch.nn as nn

class SpectralFeatureMapper(nn.Module):
    def __init__(self, eps=1e-5):
        super(SpectralFeatureMapper, self).__init__()
        self.eps = eps
        #materials_path = glob.glob('materials_numpy/*.npy')
        #materials_path = sorted(materials_path)

        #materials = np.zeros((46, 31))
        #for i,m in enumerate(materials_path):
        #    materials[i,:] = np.load(m)
        #    print(i)

        #self.members = nn.Parameter(torch.from_numpy(materials)).cuda().float()
        self.members = nn.Parameter(torch.rand(44, 31)).float()

    def forward(self, data):
        """
        Args
        ----
        data: torch.Tensor
            A tensor of shape (batch, channels, height, width)
        """
        # Normalize the members
        m = self.members / (torch.sqrt(torch.einsum('ij,ij->i', self.members, self.members))[:, None] + self.eps)

        # Reshape data to (batch, height*width, channels) for easier computation
        batch, channels, height, width = data.shape
        data = data.contiguous().view(batch, channels, height * width).permute(0, 2, 1)

        # Compute norms with epsilon
        norms = torch.sqrt(torch.einsum('bij,bij->bi', data, data)) + self.eps

        # Compute dot products
        dots = torch.einsum('bij,mj->bim', data, m)

        # Apply softmax to angles
        dots = torch.clamp(dots / norms[:, :, None], -1 + self.eps, 1 - self.eps)

        # Compute angles and permute back to (batch, members, height, width)
        angles = -torch.acos(dots).permute(0, 2, 1).contiguous().view(batch, -1, height, width)

        return angles
