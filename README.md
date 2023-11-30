# ComfyUI-Advanced-ControlNet
Nodes for scheduling ControlNet strength across timesteps and batched latents, as well as applying custom weights and attention masks. The ControlNet nodes here fully support sliding context sampling, like the one used in the  [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) nodes. Currently supports ControlNets, T2IAdapters, and ControlLoRAs. Kohya Controllllite support coming soon.

Custom weights allow replication of the "My prompt is more important" feature of Auto1111's sd-webui ControlNet extension.

ControlNet preprocessors are available through [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) nodes

## Features
- Timestep and latent strength scheduling
- Attention masks
- Soft weights to replicate "My prompt is more important" feature from sd-webui ControlNet extension, and also change the scaling.
- ControlNet, T2IAdapter, and ControlLoRA support for sliding context windows.

## Table of Contents:
- [Scheduling Explanation](#scheduling-explanation)
- [Nodes](#nodes)
- [Usage](#usage)


# Scheduling Explanation

The two core concepts for scheduling are ***Timestep Keyframes*** and ***Latent Keyframes***.

***Timestep Keyframes*** hold the values that guide the settings for a controlnet, and begin to take effect based on their start_percent, which corresponds to the percentage of the sampling process. They can contain masks for the strengths of each latent, control_net_weights, and latent_keyframes (specific strengths for each latent), all optional.

***Latent Keyframes*** determine the strength of the controlnet for specific latents - all they contain is the batch_index of the latent, and the strength the controlnet should apply for that latent. As a concept, latent keyframes achieve the same affect as a uniform mask with the chosen strength value.

![advcn_image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/e6275264-6c3f-4246-a319-111ee48f4cd9)

# Nodes

The ControlNet nodes provided here are the ***Apply Advanced ControlNet*** and ***Load Advanced ControlNet Model*** (or diff) nodes. The vanilla ControlNet nodes are also compatible, and can be used almost interchangeably - the only difference is that **at least one of these nodes must be used** for Advanced versions of ControlNets to be used (important for sliding context sampling, like with AnimateDiff-Evolved).

Key:

- 🟩 - required inputs
- 🟨 - optional inputs
- 🟥 - optional input/output, but not recommended to use unless needed
- 🟦 - start as widgets, can be converted to inputs

## Apply Advanced ControlNet
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/dc541d41-70df-4a71-b832-efa65af98f06)

Same functionality as the vanilla Apply Advanced ControlNet (Advanced) node, except with Advanced ControlNet features added to it. Automatically converts any ControlNet from ControlNet loaders into Advanced versions.

### Inputs
- 🟩***positive***: conditioning (positive).
- 🟩***negative***: conditioning (negative).
- 🟩***control_net***: loaded controlnet; will be converted to Advanced version automatically by this node, if it's a supported type.
- 🟩***image***: images to guide controlnets - if the loaded controlnet requires it, they must preprocessed images. If one image provided, will be used for all latents. If more images provided, will use each image separately for each latent. If not enough images to meet latent count, will repeat the images from the beginning to match vanilla ControlNet functionality.
- 🟨***mask_optional***: attention masks to apply to controlnets; basically, decides what part of the image the controlnet to apply to (and the relative strength, if the mask is not binary). Same as image input, if you provide more than one mask, each can apply to a different latent.
- 🟨***timestep_kf***: timestep keyframes to guide controlnet effect throughout sampling steps.
- 🟨***latent_kf_override***: override for latent keyframes, useful if no other features from timestep keyframes is needed. *NOTE: this latent keyframe will be applied to ALL timesteps, regardless if there are other latent keyframes attached to connected timestep keyframes.*
- 🟨***weights_override***: override for weights, useful if no other features from timestep keyframes is needed. *NOTE: this weight will be applied to ALL timesteps, regardless if there are other weights attached to connected timestep keyframes.*
- 🟦***strength***: strength of controlnet; 1.0 is full strength, 0.0 is no effect at all.
- 🟦***start_percent***: sampling step percentage at which controlnet should start to be applied - no matter what start_percent is set on timestep keyframes, they won't take effect until this start_percent is reached.
- 🟦***stop_percent***: sampling step percentage at which controlnet should stop being applied - no matter what start_percent is set on timestep keyframes, they won't take effect once this end_percent is reached.

### Outputs
- ***positive***: conditioning (positive) with applied controlnets
- ***negative***: conditioning (negative) with applied controlnets

## Load Advanced ControlNet Model
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/4a7f58a9-783d-4da4-bf82-bc9c167e4722)

Loads a ControlNet model and converts it into an Advanced version that supports all the features in this repo. When used with **Apply Advanced ControlNet** node, there is no reason to use the timestep_keyframe input on this node - use timestep_kf on the Apply node instead.

### Inputs
- 🟥***timestep_keyframe***: optional and likely unnecessary input to have ControlNet use selected timestep_keyframes - should not be used unless you need to. Useful if this node is not attached to **Apply Advanced ControlNet** node, but still want to use Timestep Keyframe, or to use TK_SHORTCUT outputs from ControlWeights in the same scenario. Will be overriden by the timestep_kf input on **Apply Advanced ControlNet** node, if one is provided there.
- 🟨***model***: model to plug into the diff version of the node. Some controlnets are designed for receive the model; if you don't know what this does, you probably don't want tot use the diff version of the node.

### Outputs

## Workflows

### AnimateDiff Workflows
***Latent Keyframes*** identify which latents in a batch the ControlNet should apply to, and at what strength. They connect to a ***Timestep Keyframe*** to identify at what point in the generation to kick in (for basic use, start_percent on the Timestep Keyframe should be 0.0). Latent Keyframe nodes can be chained to apply the ControlNet to multiple keyframes at various strengths.

