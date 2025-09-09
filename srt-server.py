import io
import soundfile as sf
import sys
import tempfile
import torch
import subprocess
import numpy as np

from abc import ABC, abstractmethod
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Union
from urllib.parse import unquote

app = FastAPI()

class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, file, audio_content, **kwargs):
        pass

class TransformersEngine(TranscriptionEngine):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        # --- GŁÓWNA ZMIANA: Używamy nowego, lepszego modelu po polsku ---
        model_id = "Aspik101/distil-whisper-large-v3-pl"

        # Usunęliśmy argument `attn_implementation`, ponieważ nie jest zalecany w przykładzie dla tego modelu
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        # Zaktualizowaliśmy parametry pipeline zgodnie z zaleceniami dla Distil-Whisper
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,  # Optymalna wartość dla tego modelu
            batch_size=16,
            return_timestamps=False, # Zostawiamy na False, aby uniknąć błędów
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, file, audio_content, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", {})
        result = self.pipe(audio_content, generate_kwargs=generate_kwargs)
        return result["text"], result.get("chunks", [])

# Upewniamy się, że używamy naszego silnika
engine = TransformersEngine()
logger.info(f"Using TransformersEngine with model: {engine.pipe.model.config.name_or_path}")


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
    temperature_inc: float = Form(0.0),
    response_format: str = Form("json")
):
    audio_content = await file.read()
    
    # Dla modelu dedykowanego językowi polskiemu, parametr 'language' nie jest już konieczny,
    # ale jego obecność nie zaszkodzi.
    transcribe_args = {
        "temperature": temperature + temperature_inc,
    }
    
    text, segments = engine.transcribe(
        file,
        audio_content,
        generate_kwargs=transcribe_args
    )
    
    response = {"text": text}
    return JSONResponse(content=response, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)