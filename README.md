# FLUX.1-Dev LoRA Explorer Cog Model

This is an implementation of [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a [Cog](https://github.com/replicate/cog) model.

Named LoRA Explorer, to explore the model with different LoRA weights.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="a beautiful castle frstingln illustration" -hf_lora="alvdansen/frosting_lane_flux"
    cog predict -i prompt="A bohemian-style female travel blogger with sun-kissed skin and messy beach waves" -i control_type="pose" -i control_image=@openpose.jpg

![Output](output.0.png)
