from pytorch_toolbelt.modules import encoders, decoders
from pytorch_toolbelt.modules.heads import ResizeHead
from .core import GenericSegmentationModel



def hrnet_fpn() -> GenericSegmentationModel:
    encoder = encoders.HRNetW18Encoder(
        pretrained=True, layers=[1, 2, 3, 4], use_incre_features=False
    )
    encoder.change_input_channels(4)
    decoder = decoders.FPNDecoder(
        input_spec=encoder.get_output_spec(),
        out_channels=256
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

if __name__ == '__main':
    model = hrnet_fpn()
    print(model)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))