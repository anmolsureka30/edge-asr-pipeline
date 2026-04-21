# edge_audio_intelligence/modules/diarization/pyannote_diarizer.py
import os
import torch
import numpy as np
from pyannote.audio import Pipeline
from .base import BaseDiarizer

class PyannoteDiarizer(BaseDiarizer):
    def __init__(self, name="pyannote_diarizer", model_name="pyannote/speaker-diarization-3.1", use_auth_token=None):
        super().__init__(name=name)
        
        # Save the model name so get_config can report it
        self.model_name = model_name 
        
        token = use_auth_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Hugging Face token is required. Set the 'HF_TOKEN' environment variable.")
        
        print(f"[{self.name}] Loading Pyannote pipeline: {model_name}...")
        
        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def get_config(self) -> dict:
        """Return module configuration for the pipeline logger."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "sample_rate": getattr(self, 'sample_rate', 16000)
        }

    def diarize(self, audio: np.ndarray, sample_rate: int):
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        
        # Run the diarization pipeline
        output = self.pipeline({"waveform": audio_tensor, "sample_rate": sample_rate})
        
        # FIX: Handle pyannote API changes (older versions vs 4.x+)
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization  # v4.x extracts it from the dataclass
        else:
            annotation = output                      # Older versions return it directly
        
        segments = []
        # We can now safely run itertracks on the annotation object
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
            
        return segments