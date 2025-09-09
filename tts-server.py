from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from TTS.api import TTS
import librosa
import numpy as np
import io
import time
import re
import soundfile as sf
import torch

app = FastAPI()

print('Loading VITS model for Polish...')
t0 = time.time()
vits_model = 'tts_models/pl/mai_female/vits'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tts_vits = TTS(vits_model).to(device)
elapsed = time.time() - t0
print(f"Loaded in {elapsed:.2f}s")

# Usunęliśmy zbędny parametr 'speaker'
class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Przetwarzanie tekstu - usunęliśmy linię kasującą polskie znaki
        text = request.text.strip()

        t0 = time.time()
        # Wywołujemy syntezę bez zbędnego parametru 'speaker'
        wav_np = tts_vits.tts(text)
        generation_time = time.time() - t0

        # Używamy poprawnej częstotliwości próbkowania z modelu
        original_sr = tts_vits.synthesizer.output_sample_rate
        audio_duration = len(wav_np) / original_sr
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s (RTF: {rtf:.2f})")

        wav_np = np.array(wav_np)
        wav_np = np.clip(wav_np, -1, 1)

        # Resample do 24kHz dla formatu Opus
        wav_np_24k = librosa.resample(wav_np, orig_sr=original_sr, target_sr=24000)

        # Konwersja do Opus
        buffer = io.BytesIO()
        sf.write(buffer, wav_np_24k, 24000, format='ogg', subtype='opus')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/ogg; codecs=opus")
    except Exception as e:
        print(f"Error in TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)