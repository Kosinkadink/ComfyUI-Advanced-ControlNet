import torch
import torch.nn as nn
from torch import Tensor

import comfy.model_detection
from comfy.utils import UNET_MAP_BASIC, UNET_MAP_RESNET, UNET_MAP_ATTENTIONS, TRANSFORMER_BLOCKS

import torch


from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from comfy.ldm.modules.attention import SpatialVideoTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, VideoResBlock, Downsample 
from comfy.ldm.util import exists
import comfy.ops


class SVDControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=torch.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_spatial_context=False,
        extra_ff_mix_layer=False,
        merge_strategy="fixed",
        merge_factor=0.5,
        video_kernel_size=3,
        device=None,
        operations=comfy.ops.disable_weight_init,
        **kwargs,
    ):
        super().__init__()
        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))

        transformer_depth = transformer_depth[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels, operations=operations, dtype=self.dtype, device=device)])

        self.input_hint_block = TimestepEmbedSequential(
                    operations.conv_nd(dims, hint_channels, 16, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 16, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 32, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 32, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 96, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 96, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 256, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 256, model_channels, 3, padding=1, dtype=self.dtype, device=device)
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    VideoResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                        video_kernel_size=video_kernel_size,
                        merge_strategy=merge_strategy, merge_factor=merge_factor,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialVideoTransformer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                checkpoint=use_checkpoint, dtype=self.dtype, device=device, operations=operations,
                                use_spatial_context=use_spatial_context, ff_in=extra_ff_mix_layer,
                                merge_strategy=merge_strategy, merge_factor=merge_factor,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        VideoResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                            video_kernel_size=video_kernel_size,
                            merge_strategy=merge_strategy, merge_factor=merge_factor,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            VideoResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
                video_kernel_size=video_kernel_size,
                merge_strategy=merge_strategy, merge_factor=merge_factor,
            )]
        if transformer_depth_middle >= 0:
            mid_block += [SpatialVideoTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            checkpoint=use_checkpoint, dtype=self.dtype, device=device, operations=operations,
                            use_spatial_context=use_spatial_context, ff_in=extra_ff_mix_layer,
                            merge_strategy=merge_strategy, merge_factor=merge_factor,
                        ),
            VideoResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
                video_kernel_size=video_kernel_size,
                merge_strategy=merge_strategy, merge_factor=merge_factor,
            )]
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self.middle_block_out = self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device)
        self._feature_size += ch

    def make_zero_conv(self, channels, operations=None, dtype=None, device=None):
        return TimestepEmbedSequential(operations.conv_nd(self.dims, channels, channels, 1, padding=0, dtype=dtype, device=device))

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        cond = kwargs["cond"]
        num_video_frames = cond["num_video_frames"]
        image_only_indicator = cond.get("image_only_indicator", None)
        time_context = cond.get("time_context", None)
        del cond

        guided_hint = self.input_hint_block(hint, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)

        out_output = []
        out_middle = []

        hs = []
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            out_output.append(zero_conv(h, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator))

        h = self.middle_block(h, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        out_middle.append(self.middle_block_out(h, emb, context, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator))

        return {"middle": out_middle, "output": out_output}


TEMPORAL_TRANSFORMER_BLOCKS = {
    "norm_in.weight",
    "norm_in.bias",
    "ff_in.net.0.proj.weight",
    "ff_in.net.0.proj.bias",
    "ff_in.net.2.weight",
    "ff_in.net.2.bias",
}
TEMPORAL_TRANSFORMER_BLOCKS.update(TRANSFORMER_BLOCKS)


TEMPORAL_UNET_MAP_ATTENTIONS = {
    "time_mixer.mix_factor",
}
TEMPORAL_UNET_MAP_ATTENTIONS.update(UNET_MAP_ATTENTIONS)


TEMPORAL_TRANSFORMER_MAP = {
    "time_pos_embed.0.weight": "time_pos_embed.linear_1.weight",
    "time_pos_embed.0.bias": "time_pos_embed.linear_1.bias",
    "time_pos_embed.2.weight": "time_pos_embed.linear_2.weight",
    "time_pos_embed.2.bias": "time_pos_embed.linear_2.bias",
}


TEMPORAL_RESNET = {
     "time_mixer.mix_factor",
}


def svd_unet_config_from_diffusers_unet(state_dict: dict[str, Tensor], dtype):
    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = comfy.model_detection.count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = comfy.model_detection.count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = comfy.model_detection.count_blocks(state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            transformer_depth.append(0)
            transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    # based on unet_config of SVD
    SVD = {
        'use_checkpoint': False,
        'image_size': 32,
        'use_spatial_transformer': True,
        'legacy': False,
        'num_classes': 'sequential',
        'adm_in_channels': 768,
        'dtype': dtype,
        'in_channels': 8,
        'out_channels': 4,
        'model_channels': 320,
        'num_res_blocks': [2, 2, 2, 2],
        'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
        'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'channel_mult': [1, 2, 4, 4],
        'transformer_depth_middle': 1,
        'use_linear_in_transformer': True,
        'context_dim': 1024,
        'extra_ff_mix_layer': True,
        'use_spatial_context': True,
        'merge_strategy': 'learned_with_images',
        'merge_factor': 0.0,
        'video_kernel_size': [3, 1, 1],
        'use_temporal_attention': True,
        'use_temporal_resblock': True,
        'num_heads': -1,
        'num_head_channels': 64,
        }

    supported_models = [SVD]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return comfy.model_detection.convert_config(unet_config)
    return None


def svd_unet_to_diffusers(unet_config):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, b)] = "input_blocks.{}.0.{}".format(n, b)
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.time_stack.{}".format(n, b)
                #diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in TEMPORAL_UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for b in TEMPORAL_TRANSFORMER_MAP:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, TEMPORAL_TRANSFORMER_MAP[b])] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
        for b in TEMPORAL_TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.temporal_transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.time_stack.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in TEMPORAL_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, b)] = "middle_block.{}.{}".format(n, b)
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.spatial_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.temporal_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.time_stack.{}".format(n, b)
            #diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map
