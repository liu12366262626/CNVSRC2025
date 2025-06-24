# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# NeuralNets

from . import layers
from . import modules
from . import blocks
from . import attentions
from . import normalizations

###############################################################################
# Networks
###############################################################################


class ConformerInterCTC(nn.Module):
    def __init__(
        self,
        dim_model,
        num_blocks,
        interctc_blocks,
        vocab_size,
        loss_prefix="ctc",
        att_params={"class": "MultiHeadAttention", "num_heads": 4},
        conv_params={
            "class": "Conv1d",
            "params": {"padding": "same", "kernel_size": 31},
        },
        ff_ratio=4,
        drop_rate=0.1,
        pos_embedding=None,
        mask=None,
        conv_stride=1,
        batch_norm=True,
    ):
        super(ConformerInterCTC, self).__init__()

        # Inter CTC Params
        self.interctc_blocks = interctc_blocks
        self.loss_prefix = loss_prefix

        # Single Stage
        if isinstance(dim_model, int):
            dim_model = [dim_model]
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Conformer Stages
        i = 1
        self.conformer_blocks = nn.ModuleList()
        self.interctc_modules = nn.ModuleList()
        self.diff_step_layers = nn.ModuleList()
        for stage_id in range(len(num_blocks)):
            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):
                # Transposed Block
                transposed_block = "Transpose" in conv_params["class"]

                # Downsampling Block
                down_block = (
                    ((block_id == 0) and (stage_id > 0))
                    if transposed_block
                    else (
                        (block_id == num_blocks[stage_id] - 1)
                        and (stage_id < len(num_blocks) - 1)
                    )
                )

                # Block
                self.conformer_blocks.append(blocks.ConformerBlock(
                        dim_model=dim_model[stage_id - (1 if transposed_block and down_block else 0)],
                        dim_expand=dim_model[stage_id + (1 if not transposed_block and down_block else 0)],
                        ff_ratio=ff_ratio,
                        drop_rate=drop_rate,
                        att_params=att_params[stage_id - (1 if transposed_block and down_block else 0)] if isinstance(att_params, list) else att_params,
                        conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                        conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                        batch_norm=batch_norm
                ))

                # the layer-specific fc for diffusion step embedding
                diff_dim = dim_model[stage_id - (1 if transposed_block and down_block else 0)] # self.conformer_blocks[-1].norm.weight.shape[0]
                self.diff_step_layers.append(nn.Linear(512, diff_dim))

                # InterCTC Block
                if i in interctc_blocks:
                    self.interctc_modules.append(
                        modules.InterCTCResModule(
                            dim_model=dim_model[
                                stage_id
                                + (1 if not transposed_block and down_block else 0)
                            ],
                            vocab_size=vocab_size,
                        )
                    )

                i += 1

    def forward(self, x, lengths, diffusion_step_embed):
        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)

        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Conformer Blocks
        interctc_outputs = {}
        j = 0
        for i, block in enumerate(self.conformer_blocks):
            # Diffusion step
            B = x.size(0)
            part_t = self.diff_step_layers[i](diffusion_step_embed)
            part_t = part_t.view([B, 1, -1])
            x = x + part_t

            # Conformer Block
            x = block(x, mask=mask)

            # InterCTC Block , Nothing to do
            if i + 1 in self.interctc_blocks:
                x, logits = self.interctc_modules[j](x)
                j += 1
                key = self.loss_prefix + "_" + str(i)
            else:
                logits = None

            # Strided Block
            if block.stride > 1:
                # Stride Mask (1 or B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, :: block.stride, :: block.stride]

                # Update Seq Lengths
                if lengths is not None:
                    lengths = (
                        torch.div(lengths - 1, block.stride, rounding_mode="floor") + 1
                    )

            if logits != None:
                interctc_outputs[key] = [logits, lengths]

        return x, lengths, interctc_outputs


class AudioEfficientConformerEncoder(nn.Module):
    def __init__(self, include_head=True, vocab_size=256, att_type="patch", interctc_blocks=[3, 6, 10, 13], num_blocks=[5, 6, 5], loss_prefix="ctc"):
        super(AudioEfficientConformerEncoder, self).__init__()

        assert att_type in ["regular", "grouped", "patch"]

        # Params
        n_mels = 80
        kernel_size = 15
        drop_rate = 0.1
        attn_drop_rate = 0.0
        max_pos_encoding = 10000
        subsampling_filters = 180
        dim_model = [180, 256, 360]
        num_heads = 4

        # Unsqueeze (B, N, T) -> (B, 1, N, T)
        self.unsqueeze = layers.Unsqueeze(dim=1)

        # Stem (B, 1, N, T) -> (B, C, N', T')
        self.subsampling_module = modules.ConvNeuralNetwork(
            dim_input=1,
            dim_layers=subsampling_filters,
            kernel_size=3,
            strides=2,
            norm="BatchNorm2d",
            act_fun="Swish",
            drop_rate=0.0,
            dim=2,
        )

        # Reshape (B, C, N, T) -> (B, D, T)
        self.reshape = layers.Reshape(
            shape=(subsampling_filters * n_mels // 2, -1), include_batch=False
        )

        # Transpose (B, D, T) -> (B, T, D)
        self.transpose = layers.Transpose(1, 2)

        # Linear Proj
        self.linear = layers.Linear(subsampling_filters * n_mels // 2, dim_model[0])

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(512, subsampling_filters)

        # Conformer
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params=[
                {
                    "class": "RelPosPatch1dMultiHeadAttention",
                    "params": {
                        "num_heads": num_heads,
                        "patch_size": 3,
                        "attn_drop_rate": attn_drop_rate,
                        "num_pos_embeddings": max_pos_encoding,
                        "weight_init": "default",
                        "bias_init": "default",
                    },
                },
                {
                    "class": "RelPos1dMultiHeadAttention",
                    "params": {
                        "num_heads": num_heads,
                        "attn_drop_rate": attn_drop_rate,
                        "num_pos_embeddings": max_pos_encoding,
                        "weight_init": "default",
                        "bias_init": "default",
                    },
                },
                {
                    "class": "RelPos1dMultiHeadAttention",
                    "params": {
                        "num_heads": num_heads,
                        "attn_drop_rate": attn_drop_rate,
                        "num_pos_embeddings": max_pos_encoding,
                        "weight_init": "default",
                        "bias_init": "default",
                    },
                },
            ],
            conv_params={
                "class": "Conv1d",
                "params": {"padding": "same", "kernel_size": kernel_size},
            },
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None,
            mask=attentions.Mask(),
            conv_stride=2,
            batch_norm=True,
            loss_prefix=loss_prefix,
        )

        # Head
        self.head = (
            layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()
        )

    def forward(self, x, lengths, diffusion_step_embed):

        # Unsqueeze
        x = self.unsqueeze(x)

        # Stem
        x, lengths = self.subsampling_module(x, lengths)

        # Diffusion step
        B = x.size(0)
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, -1, 1, 1])
        x = x + part_t

        # Reshape
        x = self.reshape(x)

        # Transpose
        x = self.transpose(x)

        # Linear Proj
        x = self.linear(x)

        # Conformer
        x, lengths, interctc_outputs = self.back_end(x, lengths, diffusion_step_embed)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs


