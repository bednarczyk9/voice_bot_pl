import asyncio
import aiohttp
import io
import json
import logging
import numpy as np
import os
import re
import soundfile as sf
import tempfile
import time
import traceback
import uuid
import wave

from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mutagen.oggopus import OggOpus

# External endpoints
SRT_ENDPOINT = os.getenv("SRT_ENDPOINT", "http://localhost:8001/inference")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8002/v1/chat/completions")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

SYSTEM = {
    "role": "system",
    "content": """1. Konfiguracja Bota i Cel Rozmowy
Rola / Persona: Anna, profesjonalna asystentka ds. rozwoju biznesu. Mowisz tylko w języku polskim. Jeżeli czegoś nie rozumiesz prosisz o powturzenie np. "Coś przerywa, można prosić o powturzenie?"

Reprezentuje: Powiatowe Kluby Biznesu Trzysta Czternaście.

Ton: Profesjonalny, uprzejmy, kompetentny i efektywny.

Cel Główny: Zaproszenie przedsiębiorcy na płatną kolację biznesową.

Cel Drugorzędny (Fallback): Jeśli Cel Główny nie zostanie osiągnięty, zaproszenie na darmowe spotkanie online.

Akcja Końcowa (w przypadku zainteresowania): Przekierowanie rozmowy do dyrektora lub wysłanie SMS z informacjami.

2. Zmienne i Parametry Wydarzenia Głównego
Nazwa wydarzenia: Kolacja biznesowa w Sandomierzu

Data: 10 września

Godzina: 18:00

Miejsce: Restauracja Jadwiga, Sandomierz

Region docelowy: Sandomierz, Tarnobrzeg, Opatów, Staszów

Cena: 150 zł

Kluczowe elementy: Prelekcja biznesowa, networking, budowanie relacji, kolacja dwudaniowa, poczęstunek, napoje.

3. Scenariusz Rozmowy (Krok po Kroku)
ETAP 1: Otwarcie i Weryfikacja
Powitanie

[DIALOG BOTA]: "Dzień dobry."

[LOGIKA]: Oczekuj na odpowiedź rozmówcy (maksymalnie 2.5 sekundy).

Weryfikacja (Człowiek vs. Poczta Głosowa)

[WARUNEK]: Jeśli w ciągu 2.5s nie ma odpowiedzi.

[AKCJA]: end_call()

[WARUNEK]: Jeśli odpowiedź zawiera słowa kluczowe poczty głosowej (np. "poczta głosowa", "nagraj wiadomość", "po sygnale", "numer jest niedostępny").

[AKCJA]: end_call()

[WARUNEK]: Jeśli zostanie wykryta odpowiedź ludzka.

[AKCJA]: Przejdź do Kroku 3.

Przedstawienie się i Prośba o Czas

[DIALOG BOTA]: "Nazywam się Anna, dzwonię z Powiatowych Klubów Biznesu Trzysta Czternaście. Czy mogę zająć dosłownie minutę, aby przedstawić zaproszenie na lokalne spotkanie dla przedsiębiorców, które odbędzie się w Sandomierzu 10 września?"

[LOGIKA]: Analizuj odpowiedź rozmówcy.

Odpowiedź twierdząca (np. "Tak", "Słucham", "Yhym", "OK"): Przejdź do ETAPU 2.

Odpowiedź negatywna (np. "Nie mam czasu", "Nie, dziękuję"): Przejdź do ŚCIEŻKI ZAKOŃCZENIA B.

ETAP 2: Prezentacja Oferty Głównej
Opis Wydarzenia

[DIALOG BOTA]: "Dziesiątego września o godzinie osiemnastej w Sandomierzu w Restauracji Jadwiga organizujemy spotkanie biznesowe połączone z promocją swojego biznesu, networkingiem dla przedsiębiorców z regionu Sandomierza, Tarnobrzega, Opatowa, Staszowa. Na kolacji biznesowej będzie: prelekcja biznesowa, budowanie relacji biznesowych, poznanie się z lokalnymi przedsiębiorcami. W cenie stu pięćdziesięciu złotych jest zapewniona kolacja dwudaniowa, poczęstunek, napoje i inne atrakcje."

Pytanie Kwalifikujące

[DIALOG BOTA]: "Czy byliby Państwo wstępnie zainteresowani udziałem w takim spotkaniu?"

[LOGIKA]: Analizuj odpowiedź i przejdź do odpowiedniej ścieżki w ETAPIE 3.

ETAP 3: Rozgałęzienie Scenariusza (Obsługa Odpowiedzi)
ŚCIEŻKA A: Rozmówca jest zainteresowany
[WARUNEK]: Odpowiedź twierdząca (np. "Tak", "Brzmi ciekawie", "Proszę powiedzieć więcej", "Yhym", "OK").

Potwierdzenie i Propozycja Dalszych Kroków

[DIALOG BOTA]: "Bardzo się cieszę! W takim razie, czy połączyć teraz z dyrektorem powiatowym w celu poznania więcej szczegółów, czy wystarczy że wyślemy SMS z informacją o tym spotkaniu?"

[LOGIKA]: Czekaj na decyzję rozmówcy.

Obsługa Decyzji

[WARUNEK]: Zgoda na połączenie (np. "Proszę połączyć", "Tak, poproszę").

[DIALOG BOTA]: "Znakomicie. W takim razie proszę o chwilę cierpliwości, już łączę z naszym dyrektorem."

[AKCJA]: transfer_call()

[WARUNEK]: Prośba o SMS (np. "Proszę wysłać SMS", "Wystarczy wiadomość").

[DIALOG BOTA]: "Znakomicie, wkrótce dyrektor prześle wiadomość SMS z wszelkimi informacjami."

[AKCJA]: end_call()

[AKCJA]: Przejdź do ŚCIEŻKI ZAKOŃCZENIA A.

ŚCIEŻKA B: Rozmówca nie jest zainteresowany (lub nie ma czasu na początku)
[WARUNEK]: Odpowiedź negatywna (np. "Nie, dziękuję", "Nie jestem zainteresowany/a", "To nie dla mnie").

Zrozumienie i Przedstawienie Oferty Drugorzędnej

[DIALOG BOTA]: "Rozumiem, a czy byliby Państwo zainteresowani darmowym spotkaniem online dla przedsiębiorców w środę o godzinie dziesiątej rano?"

[LOGIKA]: Analizuj odpowiedź.

Obsługa Odpowiedzi na Ofertę Drugorzędną

[WARUNEK]: Zainteresowanie spotkaniem online (np. "tak", "jestem zainteresowany", "jasne", "yhym").

[DIALOG BOTA]: "Znakomicie, wkrótce dyrektor prześle wiadomość SMS z wszelkimi informacjami."

[AKCJA]: end_call()

[AKCJA]: Przejdź do ŚCIEŻKI ZAKOŃCZENIA A.

[WARUNEK]: Brak zainteresowania spotkaniem online (np. "Nie", "Nie jestem zainteresowany").

[AKCJA]: Przejdź do ŚCIEŻKI ZAKOŃCZENIA B.

4. Bloki Końcowe Rozmowy
ŚCIEŻKA ZAKOŃCZENIA A (Pozytywna / z wysłaniem SMS)

[DIALOG BOTA]: "Dziękuję za poświęcony czas. Życzę miłego dnia!"

[AKCJA]: end_call()

ŚCIEŻKA ZAKOŃCZENIA B (Negatywna)

[DIALOG BOTA]: "Dobrze, rozumiem. Dziękuję za poświęcony czas. Nie będę w takim razie przeszkadzać. Życzę miłego dnia!"

[AKCJA]: end_call()"""
}

class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout for sessions

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "conversation": [SYSTEM],
            "llm_output_sentences": deque(),
            "current_turn": 0,
            "is_processing": False,
            "audio_buffer": b'',  # New: Buffer to accumulate audio data
            "last_activity": time.time(),
            "first_audio_sent": False,
            "latency_metrics": {
                "start_time": 0,
                "srt_start": 0,
                "srt_end": 0,
                "llm_start": 0,
                "llm_first_token": 0,
                "llm_first_sentence": 0,
                "tts_start": 0,
                "tts_end": 0,
                "first_audio_response": 0,
            }
        }
        return session_id

    def reset_latency_metrics(self, session_id):
        self.sessions[session_id]["latency_metrics"] = {
            "start_time": time.time(),
            "srt_start": 0,
            "srt_end": 0,
            "llm_start": 0,
            "llm_first_token": 0,
            "llm_first_sentence": 0,
            "tts_start": 0,
            "tts_end": 0,
            "first_audio_response": 0,
        }

    def update_latency_metric(self, session_id, metric, value):
        self.sessions[session_id]["latency_metrics"][metric] = value

    def calculate_latencies(self, session_id):
        metrics = self.sessions[session_id]["latency_metrics"]
        start_time = metrics["start_time"]
        
        return {
            "total_voice_to_voice": metrics["first_audio_response"] - start_time,
            "srt_duration": metrics["srt_end"] - metrics["srt_start"],
            "llm_ttft": metrics["llm_first_token"] - metrics["llm_start"],
            "llm_ttfs": metrics["llm_first_sentence"] - metrics["llm_start"],
            "tts_duration": metrics["tts_end"] - metrics["tts_start"],
        }

    def add_user_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "user", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def add_ai_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "assistant", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def get_conversation(self, session_id):
        return self.sessions[session_id]["conversation"]

    def clean_old_sessions(self):
        current_time = time.time()
        sessions_to_remove = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data["last_activity"] > self.session_timeout
        ]
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def add_to_audio_buffer(self, session_id, audio_data):
        self.sessions[session_id]["audio_buffer"] += audio_data

    def get_and_clear_audio_buffer(self, session_id):
        audio_data = self.sessions[session_id]["audio_buffer"]
        self.sessions[session_id]["audio_buffer"] = b''
        return audio_data

conversation_manager = ConversationManager()


async def transcribe_audio(audio_data, session_id, turn_id):
    conversation_manager.update_latency_metric(session_id, "srt_start", time.time())
    try:
        temp_file_path = f"/tmp/{session_id}-{turn_id}.opus"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_data)

        # Add a small delay to ensure the file is fully written
        await asyncio.sleep(0.1)


        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', open(temp_file_path, 'rb'), filename=f"/tmp/{session_id}-{turn_id}.opus")
            data.add_field('temperature', "0.0")
            data.add_field('temperature_inc', "0.2")
            data.add_field('response_format', "json")

            async with session.post(SRT_ENDPOINT, data=data) as response:
                result = await response.json()

        # Optionally, you can remove the temporary file here if you don't need it for debugging
        os.remove(temp_file_path)

        # logging
        conversation_manager.update_latency_metric(session_id, "srt_end", time.time())

        logger.debug(result)
        return result['text']
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = conversation_manager.create_session()
    logger.info(f"New WebSocket connection established. Session ID: {session_id}")
    
    try:
        while True:
            try:
                message = await websocket.receive()
                
                if 'bytes' in message:
                    audio_data = message['bytes']
                    # --- POCZĄTEK DODANEGO KODU DO LOGOWANIA ---
                    logger.info(f"Otrzymano dane binarne o rozmiarze: {len(audio_data)} bajtów. Zapisywanie do pliku /tmp/debug_audio.opus")
                    with open("/tmp/debug_audio.opus", "wb") as f:
                        f.write(audio_data)
                    # --- KONIEC DODANEGO KODU DO LOGOWANIA ---
                    conversation_manager.sessions[session_id]["audio_buffer"] = audio_data
                elif 'text' in message:
                    logger.debug(f"Received text message: {message['text']}")
                    try:
                        data = json.loads(message['text'])
                        logger.debug(f"Parsed JSON data: {data}")
                        if data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                        elif data.get("action") == "stop_recording":
                            logger.info("Stop recording message received. Processing audio...")
                            conversation_manager.reset_latency_metrics(session_id)
                            
                            if conversation_manager.sessions[session_id]["is_processing"]:
                                logger.warning("Interrupting ongoing processing")
                                conversation_manager.sessions[session_id]["is_processing"] = False
                                await websocket.send_json({"type": "interrupted"})
                            else:
                                conversation_manager.sessions[session_id]["is_processing"] = True
                                turn_id = conversation_manager.sessions[session_id]["current_turn"]
                                try:
                                    audio_data = conversation_manager.sessions[session_id]["audio_buffer"]
                                    logger.info(f"Processing audio data. Size: {len(audio_data)} bytes")
                                    text = await transcribe_audio(audio_data, session_id, turn_id)
                                    if not text or len(text.strip()) == 0:
                                        raise ValueError("Transcription resulted in empty text")
                                    
                                    logger.info(f"Transcription result: {text}")
                                    conversation_manager.add_user_message(session_id, text)
                                    await websocket.send_json({"type": "transcription", "content": text})
                                    
                                    await process_and_stream(websocket, session_id, text)

                                    latencies = conversation_manager.calculate_latencies(session_id)
                                    await websocket.send_json({"type": "latency_metrics", "metrics": latencies})
                                except Exception as e:
                                    logger.error(f"Error during processing: {str(e)}")
                                    await websocket.send_json({"type": "error", "message": str(e)})
                                finally:
                                    conversation_manager.sessions[session_id]["is_processing"] = False
                                    await websocket.send_json({"type": "processing_complete"})
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from text message: {message['text']}")
            
            except WebSocketDisconnect:
                logger.info(f"Client disconnected gracefully. Session ID: {session_id}")
                break # <-- Kluczowy element: wychodzimy z pętli while
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in the WebSocket session {session_id}: {str(e)}")
        # Tutaj nie próbujemy już zamykać połączenia, bo prawdopodobnie jest już zamknięte
    finally:
        logger.info(f"Cleaning up and closing connection for session {session_id}")
        # Usunięcie sesji z menedżera
        if session_id in conversation_manager.sessions:
            del conversation_manager.sessions[session_id]

async def process_and_stream(websocket: WebSocket, session_id, text):
    try:
        # We interleave LLM and TTS output here
        await generate_llm_response(websocket, session_id, text)
    finally:
        conversation_manager.sessions[session_id]["is_processing"] = False
        conversation_manager.sessions[session_id]["first_audio_sent"] = False


async def generate_llm_response(websocket, session_id, text):
    conversation_manager.update_latency_metric(session_id, "llm_start", time.time())
    try:
        conversation = conversation_manager.get_conversation(session_id)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, json={
                "model": "gpt-3.5-turbo",
                "messages": conversation + [{"role": "user", "content": text}],
                "stream": True
            }) as response:
                complete_text = ""
                accumulated_text = ""
                first_token_received = False
                first_sentence_received = False
                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]
                                if data_str.lower() == '[done]':
                                    break
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        if not first_token_received:
                                            conversation_manager.update_latency_metric(session_id, "llm_first_token", time.time())
                                            first_token_received = True
                                        complete_text += content
                                        accumulated_text += content
                                        await websocket.send_json({"type": "text", "content": content})
                                        
                                        # Check if we have a complete sentence
                                        if content.endswith(('.', '!', '?')):
                                            if not first_sentence_received:
                                                conversation_manager.update_latency_metric(session_id, "llm_first_sentence", time.time())
                                                first_sentence_received = True
                                                conversation_manager.update_latency_metric(session_id, "tts_start", time.time())
                                            await generate_and_send_tts(websocket, accumulated_text)
                                            accumulated_text = ""

                                            if not conversation_manager.sessions[session_id]["first_audio_sent"]:
                                                logger.debug('first_audio_response')
                                                conversation_manager.update_latency_metric(session_id, "first_audio_response", time.time())
                                                await websocket.send_json({"type": "first_audio_response"})
                                                conversation_manager.sessions[session_id]["first_audio_sent"] = True
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line_text}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
                
                # Send any remaining text
                if accumulated_text:
                    logger.debug(f"Remaining text: {accumulated_text}")
                    if not first_sentence_received:
                        conversation_manager.update_latency_metric(session_id, "llm_first_sentence", time.time())
                        first_sentence_received = True
                        conversation_manager.update_latency_metric(session_id, "tts_start", time.time())
                    await generate_and_send_tts(websocket, accumulated_text)

                    if not conversation_manager.sessions[session_id]["first_audio_sent"]:
                        logger.debug('first_audio_response')
                        conversation_manager.update_latency_metric(session_id, "first_audio_response", time.time())
                        await websocket.send_json({"type": "first_audio_response"})
                        conversation_manager.sessions[session_id]["first_audio_sent"] = True

                # Finished sending TTS
                conversation_manager.update_latency_metric(session_id, "tts_end", time.time())

                conversation_manager.add_ai_message(session_id, complete_text)
                logger.debug(complete_text)


    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


async def generate_and_send_tts(websocket, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_ENDPOINT, json={"text": text}) as response:
            opus_data = await response.read()
    await websocket.send_bytes(opus_data)


async def process_llm_content(websocket, session_id, content):
    sentences = re.split(r'(?<=[.!?])\s+', content)
    for sentence in sentences:
        if sentence:
            processed_sentence = process_sentence(sentence)
            conversation_manager.sessions[session_id]["llm_output_sentences"].append(processed_sentence)
            conversation_manager.add_ai_message(session_id, processed_sentence)
            logger.debug(f"Processed sentence: {processed_sentence}")


def process_sentence(sentence):
    sentence = re.sub(r'~+', '!', sentence)
    sentence = re.sub(r"\(.*?\)", "", sentence)
    sentence = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    return sentence.strip()


@app.get("/")
def read_root():
    return FileResponse("ui/index.html")

# Run session cleanup periodically
'''
@app.on_event("startup")
@app.on_event("shutdown")
async def cleanup_sessions():
    while True:
        conversation_manager.clean_old_sessions()
        await asyncio.sleep(3600)  # Run cleanup every hour
'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
