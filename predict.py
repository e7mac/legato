"""
Replicate Predictor for LEGATO - Optical Music Recognition Model

This module provides a Cog-compatible predictor for deploying LEGATO on Replicate.
LEGATO converts sheet music images to ABC notation or MusicXML.
"""

import tempfile
from typing import Union

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig

from legato.models import LegatoModel, LegatoConfig
from utils.convert import cleanup_abc
from utils.abc2xml import MusicXml, fixDoctype


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

        # Initialize ABC to MusicXML converter
        self.abc2xml = MusicXml()

    def predict(
        self,
        image: Path = Input(description="Sheet music image to transcribe"),
        output_format: str = Input(
            description="Output format for the transcription.",
            default="musicxml",
            choices=["abc", "musicxml"],
        ),
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
    ) -> Union[str, Path]:
        """
        Transcribe a sheet music image to ABC notation or MusicXML.

        Returns the transcription as a string (ABC) or a file path (MusicXML).
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
        abc_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if output_format == "musicxml":
            # Clean up ABC notation for conversion
            cleaned_abc = cleanup_abc(abc_output)
            if not cleaned_abc:
                xml_content = "<?xml version='1.0' encoding='utf-8'?>\n<score-partwise version=\"3.0\"></score-partwise>"
            else:
                # Convert ABC to MusicXML
                try:
                    score = self.abc2xml.parse(cleaned_abc)
                    xml_content = fixDoctype(score)
                except Exception as e:
                    # Return empty MusicXML if conversion fails
                    xml_content = f"<?xml version='1.0' encoding='utf-8'?>\n<!-- Conversion error: {str(e)} -->\n<score-partwise version=\"3.0\"></score-partwise>"

            # Write to a temporary file and return the path
            output_file = Path(tempfile.mktemp(suffix=".musicxml"))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
            return output_file

        return abc_output
