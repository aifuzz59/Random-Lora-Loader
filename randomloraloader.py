import os
import random
import hashlib
import folder_paths
import comfy.utils
import comfy.sd

class LoadRandomLora:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP", ),
                    "lora_directory": ("STRING", {
                        "default": folder_paths.get_folder_paths("loras")[0],
                        "multiline": False
                    }),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                    "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "LOADED_LORA_NAME")
    FUNCTION = "load_random_lora"

    CATEGORY = "loaders"

    def load_random_lora(self, model, clip, lora_directory, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "No LoRA applied (strength set to 0)")

        if not os.path.isdir(lora_directory):
            raise ValueError(f"Invalid LoRA directory: {lora_directory}")

        lora_files = [f for f in os.listdir(lora_directory) if f.endswith('.safetensors') or f.endswith('.pt')]
        if not lora_files:
            raise FileNotFoundError("No LoRA files found in the specified folder.")

        lora_name = random.choice(lora_files)
        lora_path = os.path.join(lora_directory, lora_name)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, lora_name)

    @classmethod
    def IS_CHANGED(cls, model, clip, lora_directory, strength_model, strength_clip):
        # Disable caching by always returning a different value
        return random.random()

NODE_CLASS_MAPPINGS = {
    "LoadRandomLora": LoadRandomLora
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRandomLora": "Load Random LoRA"
}