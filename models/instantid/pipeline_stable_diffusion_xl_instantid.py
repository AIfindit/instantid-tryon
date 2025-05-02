import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor

class StableDiffusionXLInstantIDPipeline(StableDiffusionXLPipeline):
    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale

    def encode_ip_adapter(self, image_embeds):
        if isinstance(image_embeds, list):
            image_embeds = torch.stack(image_embeds)
        return image_embeds.to(self.device)

    def _encode_prompt(self, prompt, **kwargs):
        prompt_embeds = super()._encode_prompt(prompt, **kwargs)
        if hasattr(self, "ip_adapter_image"):
            image_embeds = self.encode_ip_adapter(self.ip_adapter_image)
            prompt_embeds = prompt_embeds + self.ip_adapter_scale * image_embeds
        return prompt_embeds
