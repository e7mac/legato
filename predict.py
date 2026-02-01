"""
Replicate Predictor for LEGATO - Optical Music Recognition Model

This module provides a Cog-compatible predictor for deploying LEGATO on Replicate.
LEGATO converts sheet music images to ABC notation.
"""

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig

from legato.models import LegatoModel, LegatoConfig


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory for efficient prediction."""
        model_path = "e7mac/lega"

        # Load config and fix the vision encoder path (saved as local path during training)
        config = LegatoConfig.from_pretrained(model_path)
        config.encoder_pretrained_model_name_or_path = "e7mac/vision-encoder"

        self.model = LegatoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = self.model.to("cuda")
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Sheet music image to transcribe to ABC notation"),
        beam_size: int = Input(
            description="Beam size for generation. Higher values may improve quality but are slower.",
            default=10,
            ge=1,
            le=20,
        ),
        max_length: int = Input(
            description="Maximum length of generated ABC notation in tokens.",
            default=2048,
            ge=256,
            le=4096,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeating tokens. Higher values reduce repetition.",
            default=1.1,
            ge=1.0,
            le=2.0,
        ),
    ) -> str:
        """
        Transcribe a sheet music image to ABC notation.

        Returns the ABC notation as a string.
        """
        # Load and process the image
        img = Image.open(str(image)).convert("RGB")

        # Prepare inputs
        inputs = self.processor(images=[img], truncation=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Configure generation
        generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=beam_size,
            repetition_penalty=repetition_penalty,
        )

        # Generate ABC notation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )

        # Decode the output
        abc_output = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return abc_output[0]
