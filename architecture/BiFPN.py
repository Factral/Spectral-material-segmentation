from pytorch_toolbelt.modules import encoders, decoders, heads
from pytorch_toolbelt.modules.heads import ResizeHead
from pytorch_toolbelt.modules import ACT_RELU, ACT_SILU
from .core import GenericSegmentationModel
    
def b4_bifpn():
    encoder = encoders.TimmB4Encoder(  first_conv_stride_one=True,
        pretrained=True, layers=[1, 2, 3, 4], drop_path_rate=0.2, activation=ACT_SILU
    )
    decoder = decoders.BiFPNDecoder(
        input_spec=encoder.get_output_spec(),
        out_channels=256,
        num_layers=3,
        activation=ACT_SILU
    )
    head = ResizeHead(
        input_spec=decoder.get_output_spec(),
        num_classes=31,
        dropout_rate=0.2,
    )
    return GenericSegmentationModel(
        encoder,
        decoder,
        head,
    )

if __name__ == '__main__':
    model = b4_bifpn()
    print(model)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))