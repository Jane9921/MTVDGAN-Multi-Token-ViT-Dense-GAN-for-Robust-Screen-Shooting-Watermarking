'''
encoder-noiser-decoder
修改noiser的时候，要记得修改这里的调用

2024.8.26
替换noise 为 pimog noise
'''
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser

from model.NoiseLayer import ScreenShooting
import torch

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """

    def __init__(self, config: HiDDenConfiguration, noiser: ScreenShooting):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = ScreenShooting()

        self.decoder = Decoder(config)
        self.average = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, batch):
        image, message = batch
        encoded_image = self.encoder(image, message)
        noised_image = self.noiser(encoded_image)
        noised_image = noised_image.to(torch.float32)
        decoded_message_m = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message_m
