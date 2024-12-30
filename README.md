# ComfyUI-Advanced-ControlNet
Nodes for scheduling ControlNet strength across timesteps and batched latents, as well as applying custom weights and attention masks. The ControlNet nodes here fully support sliding context sampling, like the one used in the  [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) nodes. Currently supports ControlNets, T2IAdapters, ControlLoRAs, ControlLLLite, SparseCtrls, SVD-ControlNets, and Reference.

Custom weights allow replication of the "My prompt is more important" feature of Auto1111's sd-webui ControlNet extension via Soft Weights, and the "ControlNet is more important" feature can be granularly controlled by changing the uncond_multiplier on the same Soft Weights.

ControlNet preprocessors are available through [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) nodes.

## Features
- Timestep and latent strength scheduling
- Attention masks
- Replicate ***"My prompt is more important"*** feature from sd-webui-controlnet extension via ***Soft Weights***, and allow softness to be tweaked via ***base_multiplier***
- Replicate ***"ControlNet is more important"*** feature from sd-webui-controlnet extension via ***uncond_multiplier*** on ***Soft Weights***
  - uncond_multiplier=0.0 gives identical results of auto1111's feature, but values between 0.0 and 1.0 can be used without issue to granularly control the setting.
- ControlNet, T2IAdapter, and ControlLoRA support for sliding context windows
- ControlLLLite support
- ControlNet++ support
- CtrLoRA support
  - Relevant models linked on [CtrLoRA github page](https://github.com/xyfJASON/ctrlora)
- SparseCtrl support
- SVD-ControlNet support
  - Stable Video Diffusion ControlNets trained by **CiaraRowles**: [Depth](https://huggingface.co/CiaraRowles/temporal-controlnet-depth-svd-v1/tree/main/controlnet), [Lineart](https://huggingface.co/CiaraRowles/temporal-controlnet-lineart-svd-v1/tree/main/controlnet)  
- Reference support
  - Supports ```reference_attn```, ```reference_adain```, and ```refrence_adain+attn``` modes. ```style_fidelity``` and ```ref_weight``` are equivalent to style_fidelity and control_weight in Auto1111, respectively, and strength of the Apply ControlNet is the balance between ref-influenced result and no-ref result. There is also a Reference ControlNet (Finetune) node that allows adjust the style_fidelity, weight, and strength of attn and adain separately.

## Table of Contents:
- [Scheduling Explanation](#scheduling-explanation)
- [Nodes](#nodes)
- [Usage](#usage) (will fill this out soon)


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
- 🟦 - start as widgets, can be converted to inputs
- 🟥 - optional input/output, but not recommended to use unless needed
- 🟪 - output

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
- 🟪***positive***: conditioning (positive) with applied controlnets
- 🟪***negative***: conditioning (negative) with applied controlnets

## Load Advanced ControlNet Model
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/4a7f58a9-783d-4da4-bf82-bc9c167e4722)

Loads a ControlNet model and converts it into an Advanced version that supports all the features in this repo. When used with **Apply Advanced ControlNet** node, there is no reason to use the timestep_keyframe input on this node - use timestep_kf on the Apply node instead.

### Inputs
- 🟥***timestep_keyframe***: optional and likely unnecessary input to have ControlNet use selected timestep_keyframes - should not be used unless you need to. Useful if this node is not attached to **Apply Advanced ControlNet** node, but still want to use Timestep Keyframe, or to use TK_SHORTCUT outputs from ControlWeights in the same scenario. Will be overriden by the timestep_kf input on **Apply Advanced ControlNet** node, if one is provided there.
- 🟨***model***: model to plug into the diff version of the node. Some controlnets are designed for receive the model; if you don't know what this does, you probably don't want tot use the diff version of the node.

### Outputs
- 🟪***CONTROL_NET***: loaded Advanced ControlNet

## Timestep Keyframe
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/404f3cfe-5852-4eed-935b-37e32493d1b5)

Scheduling node across timesteps (sampling steps) based on the set start_percent. Chaining Timestep Keyframes allows ControlNet scheduling across sampling steps (percentage-wise), through a timestep keyframe schedule.

### Inputs
- 🟨***prev_timestep_kf***: used to chain Timestep Keyframes together to create a schedule. The order does not matter - the Timestep Keyframes sort themselves automatically by their start_percent. *Any Timestep Keyframe contained in the prev_timestep_keyframe that contains the same start_percent as the Timestep Keyframe will be overwritten.*
- 🟨***cn_weights***: weights to apply to controlnet while this Timestep Keyframe is in effect. Must be compatible with the loaded controlnet, or will throw an error explaining what weight types are compatible. If inherit_missing is True, if no control_net_weight is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a weight_override, the weight_override will be used during sampling instead of control_net_weight.*
- 🟨***latent_keyframe***: latent keyframes to apply to controlnet while this Timestep Keyframe is in effect. If inherit_missing is True, if no latent_keyframe is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a latent_kf_override, the latent_lf_override will be used during sampling instead of latent_keyframe.*
- 🟨***mask_optional***: attention masks to apply to controlnets; basically, decides what part of the image the controlnet to apply to (and the relative strength, if the mask is not binary). Same as mask_optional on the Apply Advanced ControlNet node, can apply either one maks to all latents, or individual masks for each latent. If inherit_missing is True, if no mask_optional is passed in, will attempt to reuse the last-used mask_optional in the timestep keyframe schedule. It is NOT overriden by mask_optional on the Apply  Advanced ControlNet node; will be used together.
- 🟦***start_percent***: sampling step percentage at which this Timestep Keyframe qualifies to be used. Acts as the 'key' for the Timestep Keyframe in the timestep keyframe schedule.
- 🟦***strength***: strength of the controlnet; multiplies the controlnet by this value, basically, applied alongside the strength on the Apply ControlNet node. If set to 0.0 will not have any effect during the duration of this Timestep Keyframe's effect, and will increase sampling speed by not doing any work.
- 🟦***null_latent_kf_strength***: strength to assign to latents that are unaccounted for in the passed in latent_keyframes. Has no effect if no latent_keyframes are passed in, or no batch_indeces are unaccounted in the latent_keyframes for during sampling.
- 🟦***inherit_missing***: determines if should reuse values from previous Timestep Keyframes for optional values (control_net_weights, latent_keyframe, and mask_option) that are not included on this TimestepKeyframe. To inherit only specific inputs, use default inputs.
- 🟦***guarantee_steps***: when 1 or greater, even if a Timestep Keyframe's start_percent ahead of this one in the schedule is closer to current sampling percentage, this Timestep Keyframe will still be used for the specified amount of steps before moving on to the next selected Timestep Keyframe in the following step. Whether the Timestep Keyframe is used or not, its inputs will still be accounted for inherit_missing purposes.  

### Outputs
- 🟪***TIMESTEP_KF***: the created Timestep Keyframe, that can either be linked to another or into a Timestep Keyframe input.

## Timestep Keyframe Interpolation
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/9789617c-202c-4271-92a2-0909bcf9b108)

Allows to create Timestep Keyframe with interpolated strength values in a given percent range. (The first generated keyframe will have guarantee_steps=1, rest that follow will have guarantee_steps=0).

### Inputs
- 🟨***prev_timestep_kf***: used to chain Timestep Keyframes together to create a schedule. The order does not matter - the Timestep Keyframes sort themselves automatically by their start_percent. *Any Timestep Keyframe contained in the prev_timestep_keyframe that contains the same start_percent as the Timestep Keyframe will be overwritten.*
- 🟨***cn_weights***: weights to apply to controlnet while this Timestep Keyframe is in effect. Must be compatible with the loaded controlnet, or will throw an error explaining what weight types are compatible. If inherit_missing is True, if no control_net_weight is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a weight_override, the weight_override will be used during sampling instead of control_net_weight.*
- 🟨***latent_keyframe***: latent keyframes to apply to controlnet while this Timestep Keyframe is in effect. If inherit_missing is True, if no latent_keyframe is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a latent_kf_override, the latent_lf_override will be used during sampling instead of latent_keyframe.*
- 🟨***mask_optional***: attention masks to apply to controlnets; basically, decides what part of the image the controlnet to apply to (and the relative strength, if the mask is not binary). Same as mask_optional on the Apply Advanced ControlNet node, can apply either one maks to all latents, or individual masks for each latent. If inherit_missing is True, if no mask_optional is passed in, will attempt to reuse the last-used mask_optional in the timestep keyframe schedule. It is NOT overriden by mask_optional on the Apply  Advanced ControlNet node; will be used together.
- 🟦***start_percent***: sampling step percentage at which the first generated Timestep Keyframe qualifies to be used.
- 🟦***end_percent***: sampling step percentage at which the last generated Timestep Keyframe qualifies to be used.
- 🟦***strength_start***: strength of the Timestep Keyframe at start of range.
- 🟦***strength_end***: strength of the Timestep Keyframe at end of range.
- 🟦***interpolation***: the method of interpolation.
- 🟦***intervals***: the amount of keyframes to generate in total - the first will have its start_percent equal to start_percent, the last will have its start_percent equal to end_percent.
- 🟦***null_latent_kf_strength***: strength to assign to latents that are unaccounted for in the passed in latent_keyframes. Has no effect if no latent_keyframes are passed in, or no batch_indeces are unaccounted in the latent_keyframes for during sampling.
- 🟦***inherit_missing***: determines if should reuse values from previous Timestep Keyframes for optional values (control_net_weights, latent_keyframe, and mask_option) that are not included on this TimestepKeyframe. To inherit only specific inputs, use default inputs.
- 🟦***print_keyframes***: if True, will print the Timestep Keyframes generated by this node for debugging purposes.

### Outputs
- 🟪***TIMESTEP_KF***: the created Timestep Keyframe, that can either be linked to another or into a Timestep Keyframe input.

## Timestep Keyframe From List
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/9e9c23bf-6f82-4ce7-b4d1-3016fd14707d)

Allows to create Timestep Keyframe via a list of floats, such as with Batch Value Schedule from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) nodes. (The first generated keyframe will have guarantee_steps=1, rest that follow will have guarantee_steps=0).

### Inputs
- 🟨***prev_timestep_kf***: used to chain Timestep Keyframes together to create a schedule. The order does not matter - the Timestep Keyframes sort themselves automatically by their start_percent. *Any Timestep Keyframe contained in the prev_timestep_keyframe that contains the same start_percent as the Timestep Keyframe will be overwritten.*
- 🟨***cn_weights***: weights to apply to controlnet while this Timestep Keyframe is in effect. Must be compatible with the loaded controlnet, or will throw an error explaining what weight types are compatible. If inherit_missing is True, if no control_net_weight is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a weight_override, the weight_override will be used during sampling instead of control_net_weight.*
- 🟨***latent_keyframe***: latent keyframes to apply to controlnet while this Timestep Keyframe is in effect. If inherit_missing is True, if no latent_keyframe is passed in, will attempt to reuse the last-used weights in the timestep keyframe schedule. *If Apply Advanced ControlNet node has a latent_kf_override, the latent_lf_override will be used during sampling instead of latent_keyframe.*
- 🟨***mask_optional***: attention masks to apply to controlnets; basically, decides what part of the image the controlnet to apply to (and the relative strength, if the mask is not binary). Same as mask_optional on the Apply Advanced ControlNet node, can apply either one maks to all latents, or individual masks for each latent. If inherit_missing is True, if no mask_optional is passed in, will attempt to reuse the last-used mask_optional in the timestep keyframe schedule. It is NOT overriden by mask_optional on the Apply  Advanced ControlNet node; will be used together.
- 🟩***float_strengths***: a list of floats, that will correspond to the strength of each Timestep Keyframe; first will be assigned to start_percent, last will be assigned to end_percent, and the rest spread linearly between.
- 🟦***start_percent***: sampling step percentage at which the first generated Timestep Keyframe qualifies to be used.
- 🟦***end_percent***: sampling step percentage at which the last generated Timestep Keyframe qualifies to be used.
- 🟦***null_latent_kf_strength***: strength to assign to latents that are unaccounted for in the passed in latent_keyframes. Has no effect if no latent_keyframes are passed in, or no batch_indeces are unaccounted in the latent_keyframes for during sampling.
- 🟦***inherit_missing***: determines if should reuse values from previous Timestep Keyframes for optional values (control_net_weights, latent_keyframe, and mask_option) that are not included on this TimestepKeyframe. To inherit only specific inputs, use default inputs.
- 🟦***print_keyframes***: if True, will print the Timestep Keyframes generated by this node for debugging purposes.

### Outputs
- 🟪***TIMESTEP_KF***: the created Timestep Keyframe, that can either be linked to another or into a Timestep Keyframe input.

## Latent Keyframe
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/7eb2cc4c-255c-4f32-b09b-699f713fada3)

A singular Latent Keyframe, selects the strength for a specific batch_index. If batch_index is not present during sampling, will simply have no effect. Can be chained with any other Latent Keyframe-type node to create a latent keyframe schedule.

### Inputs
- 🟨***prev_latent_kf***: used to chain Latent Keyframes together to create a schedule. *If a Latent Keyframe contained in prev_latent_keyframes have the same batch_index as this Latent Keyframe, they will take priority over this node's value.*
- 🟦***batch_index***: index of latent in batch to apply controlnet strength to. Acts as the 'key' for the Latent Keyframe in the latent keyframe schedule.
- 🟦***strength***: strength of controlnet to apply to the corresponding latent.

### Outputs
- 🟪***LATENT_KF***: the created Latent Keyframe, that can either be linked to another or into a Latent Keyframe input.

## Latent Keyframe Group
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/5ce3b795-f5fc-4dc3-ae30-a4c7f87e278c)

Allows to create Latent Keyframes via individual indeces or python-style ranges.

### Inputs
- 🟨***prev_latent_kf***: used to chain Latent Keyframes together to create a schedule. *If any Latent Keyframes contained in prev_latent_keyframes have the same batch_index as a this Latent Keyframe, they will take priority over this node's version.* 
- 🟨***latent_optional***: the latents expected to be passed in for sampling; only required if you wish to use negative indeces (will be automatically converted to real values).
- 🟦***index_strengths***: string list of indeces or python-style ranges of indeces to assign strengths to. If latent_optional is passed in, can contain negative indeces or ranges that contain negative numbers, python-style. The different indeces must be comma separated. Individual latents can be specified by ```batch_index=strength```, like ```0=0.9```. Ranges can be specified by ```start_index_inclusive:end_index_exclusive=strength```, like ```0:8=strength```. Negative indeces are possible when latents_optional has an input, with a string such as ```0,-4=0.25```.
- 🟦***print_keyframes***: if True, will print the Latent Keyframes generated by this node for debugging purposes.

### Outputs
- 🟪***LATENT_KF***: the created Latent Keyframe, that can either be linked to another or into a Latent Keyframe input.

## Latent Keyframe Interpolation
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/7986c737-83b9-46bc-aab0-ae4c368df446)

Allows to create Latent Keyframes with interpolated values in a range.

### Inputs
- 🟨***prev_latent_kf***: used to chain Latent Keyframes together to create a schedule. *If any Latent Keyframes contained in prev_latent_keyframes have the same batch_index as a this Latent Keyframe, they will take priority over this node's version.*
- 🟦***batch_index_from***: starting batch_index of range, included.
- 🟦***batch_index_to***: end batch_index of range, excluded (python-style range).
- 🟦***strength_from***: starting strength of interpolation.
- 🟦***strength_to***: end strength of interpolation.
- 🟦***interpolation***: the method of interpolation.
- 🟦***print_keyframes***: if True, will print the Latent Keyframes generated by this node for debugging purposes.

### Outputs
- 🟪***LATENT_KF***: the created Latent Keyframe, that can either be linked to another or into a Latent Keyframe input.

## Latent Keyframe From List
![image](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/assets/7365912/6cec701f-6183-4aeb-af5c-cac76f5591b7)

Allows to create Latent Keyframes via a list of floats, such as with Batch Value Schedule from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) nodes.

### Inputs
- 🟨***prev_latent_kf***: used to chain Latent Keyframes together to create a schedule. *If any Latent Keyframes contained in prev_latent_keyframes have the same batch_index as a this Latent Keyframe, they will take priority over this node's version.* 
- 🟩***float_strengths***: a list of floats, that will correspond to the strength of each Latent Keyframe; the batch_index is the index of each float value in the list.
- 🟦***print_keyframes***: if True, will print the Latent Keyframes generated by this node for debugging purposes.

### Outputs
- 🟪***LATENT_KF***: the created Latent Keyframe, that can either be linked to another or into a Latent Keyframe input.

# There are more nodes to document and show usage - will add this soon! TODO
