# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import re
import cv2
import time
import torch
import subprocess
import numpy as np
from typing import List
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetInpaintPipeline, FluxControlNetModel
from controlnet_aux import CannyDetector
from huggingface_hub import hf_hub_download
from weights import WeightsDownloadCache

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
LORA_REPO_NAME = "ByteDance/Hyper-SD"
LORA_CKPT_NAME = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
CONTROLNET_MODEL_CACHE = "FLUX-CONTROLNET"

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):

    def get_allowed_dimensions(self):
        return [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        
    def get_resized_dimensions(self, image):
        width, height = image.size
        original_aspect_ratio = width / height

        # Define a function to calculate the closeness of the aspect ratio
        def aspect_ratio_difference(dim):
            w, h = dim
            return abs((w / h) - original_aspect_ratio)

        # Select the dimension pair with the closest aspect ratio to the original
        allowed_dimensions = self.get_allowed_dimensions()
        closest_dimensions = min(allowed_dimensions, key=aspect_ratio_difference)

        return closest_dimensions

    def resize_and_flatten_image(self, image):
        width, height = self.get_resized_dimensions(image)
        resized = image.resize((width, height))
        if resized.mode == 'RGBA':
            background = Image.new('RGB', resized.size, (255, 255, 255))
            background.paste(resized, mask=resized.split()[3])
            return background
        else:
            return resized

    def preprocess_images(self, image, control_image, mask):
        image = Image.open(image)
        image = self.resize_and_flatten_image(image)
        control_image = Image.open(control_image)
        control_image = self.resize_and_flatten_image(control_image)
        mask = Image.open(mask)
        mask = self.resize_and_flatten_image(mask)
        canny_processed = self.canny_detector(control_image)
        return image, canny_processed, mask

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")

        canny_controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
            torch_dtype=torch.float16,
            cache_dir=CONTROLNET_MODEL_CACHE, 
        )
        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=canny_controlnet,
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

        self.canny_detector = CannyDetector()
        self.weights_cache = WeightsDownloadCache()

        self.eight_steps_path = hf_hub_download(LORA_REPO_NAME, LORA_CKPT_NAME)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        conditioning_scale: float = Input(
            description="ControlNet strength, depth works best at 0.2, canny works best at 0.4. Recommended range is 0.3-0.8",
            default=0.5, ge=0, le=1
        ),
        image: Path = Input(
            description="The image to restyle",
            default=None
        ),
        control_image: Path = Input(
            description="The image to control the generation",
            default=None
        ),
        mask: Path = Input(
            description="The area to inpaint",
            default=None
        ),
        strength: float = Input(
            description="Img2Img strength",
            default=0.8,
            ge=0,
            le=1
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=3.5,
            ge=0,
            le=30
        ),
        enable_hyper_flux_8_step: bool = Input(
            description="Whether to use Hyper-FLUX.1-dev-8steps or not. If False, make sure to increase your number of inference steps",
            default=True
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,le=28,default=8,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation",
            default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="jpg",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=100,
            ge=0,
            le=100,
        ),
        lora_weights: str = Input(
            description="Huggingface path, or URL to the LoRA weights. Ex: alvdansen/frosting_lane_flux",
            default=None,
        ),
        lora_scale: float = Input(
            description="Scale for the LoRA weights",
            ge=0,le=1, default=0.8,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
        flux_kwargs = {}
        hyper_scale = 0.125 if enable_hyper_flux_8_step else 0.0

        if lora_weights is not None:
            joint_attention_kwargs = {"scale": lora_scale}
            flux_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
            if re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/trained_model.tar", lora_weights):
                print(f"Downloading LoRA weights from - Replicate URL: {lora_weights}")
                local_weights_cache = self.weights_cache.ensure(lora_weights)
                lora_path = os.path.join(local_weights_cache, "output/flux_train_replicate/lora.safetensors")
                self.pipe.load_lora_weights(self.eight_steps_path, adapter_name="hyper")
                self.pipe.load_lora_weights(lora_path, adapter_name="extra")
                self.pipe.set_adapters(["hyper", "extra"], adapter_weights=[hyper_scale, lora_scale])

            elif lora_weights.endswith('.safetensors'):
                print(f"Downloading LoRA weights from - safetensor URL: {lora_weights}")
                try:
                    lora_path = self.weights_cache.ensure(lora_weights, file=True)
                except Exception as e:
                    raise Exception(f"Error downloading LoRA weights from URL: {e}")
                self.pipe.load_lora_weights(lora_path)
            else:
                raise Exception(
                    f"Invalid lora, must be either a: Replicate model.tar URL, or a URL with .safetensors extension: {lora_weights}")
        else:
            flux_kwargs["joint_attention_kwargs"] = None
            self.pipe.unload_lora_weights()
            self.pipe.load_lora_weights(self.eight_steps_path, adapter_name="hyper")
            self.pipe.set_adapters(["hyper"], adapter_weights=[hyper_scale])

        image, control_image, mask = self.preprocess_images(image, control_image, mask)

        output = self.pipe(
            prompt,
            image=image,
            mask_image=mask,
            control_image=control_image,
            controlnet_conditioning_scale=conditioning_scale,
            strength=strength,
            width=image.size[0],
            height=image.size[1],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        )

        if lora_weights is not None:
            self.pipe.unload_lora_weights()

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception("NSFW content detected. Try running it again, or try a different prompt.")

        return output_paths