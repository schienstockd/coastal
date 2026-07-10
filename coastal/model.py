"""UNet architecture with probability and embedding heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out, self.pool(conv_out)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetWithEmbeddings(nn.Module):
    """UNet that takes frame(s) + metrics as input, outputs prob + embeddings."""

    def __init__(self, num_metrics=14, num_frames=1, out_channels=1, init_features=32, depth=3, embedding_dim=2):
        super().__init__()
        self.depth = depth
        self.num_metrics = num_metrics
        self.num_frames = num_frames
        self.embedding_dim = embedding_dim
        self.init_features = init_features

        in_channels = num_frames + num_metrics

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = init_features * (2 ** i)
            self.encoders.append(Encoder(in_ch, out_ch))
            in_ch = out_ch

        bottleneck_ch = init_features * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch)

        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = init_features * (2 ** i)
            in_ch = bottleneck_ch if i == depth - 1 else (out_ch * 2)
            self.decoders.append(Decoder(in_ch, out_ch))
            bottleneck_ch = out_ch

        self.prob_head = nn.Conv2d(init_features, out_channels, 1)
        self.emb_head = nn.Conv2d(init_features, embedding_dim, 1)

    def encode_decode(self, frame_and_metrics):
        """Run encoder+decoder and return shared decoder features [B, C, H, W]."""
        x = frame_and_metrics
        encoder_outputs = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            encoder_outputs.append(skip)
        x = self.bottleneck(x)
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, encoder_outputs[-(i + 1)])
        return x

    def forward(self, frame_and_metrics):
        """
        Args:
            frame_and_metrics: [B, num_frames+num_metrics, H, W]

        Returns:
            prob: [B, 1, H, W]
            emb: [B, embedding_dim, H, W]
        """
        x = self.encode_decode(frame_and_metrics)
        return self.prob_head(x), self.emb_head(x)
