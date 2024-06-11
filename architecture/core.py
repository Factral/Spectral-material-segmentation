from torch import nn
from ..SFM import SpectralFeatureMapper

class GenericSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder, head):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.sam = SpectralFeatureMapper()

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        hs = self.head(features, output_size=x.shape[-2:])
        outputs = self.sam(hs)
        return outputs, hs
