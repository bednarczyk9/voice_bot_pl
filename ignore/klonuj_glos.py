import os
import torch
from TTS.api import TTS

# --- SEKCJA KONFIGURACJI ---
# Zmień poniższe wartości według swoich potrzeb.

# Ścieżka do pliku .wav z czystą próbką głosu (rekomendowane 5-15 sekund).
PROBKA_WAV = "/home/bed9/Desktop/Robocorp/Maria_glos.wav"

# Nazwa pliku, w którym zostanie zapisany sklonowany głos.
PLIK_WYJSCIOWY_GLOSU = "Anna_PL.pth"

# Tekst, który zostanie użyty do automatycznego przetestowania sklonowanego głosu.
TEKST_DO_TESTOWANIA = "Cześć, to jest test wygenerowany moim nowym, sklonowanym głosem."

# Nazwa pliku z wygenerowaną próbką mowy po udanym klonowaniu.
PLIK_WYJSCIOWY_TESTU = "wynik_testu_klonowania.wav"

# --- KONIEC KONFIGURACJI ---


def main():
    """
    Główna funkcja skryptu.
    """
    print("--- Rozpoczynam proces klonowania głosu ---")
    
    # Sprawdzenie, czy plik z próbką istnieje
    if not os.path.exists(PROBKA_WAV):
        print(f"!!! BŁĄD: Plik z próbką głosu nie istnieje pod ścieżką: {PROBKA_WAV}")
        return

    # Wybór urządzenia (GPU, jeśli dostępne)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Używane urządzenie: {device}")

    print("-> Ładowanie modelu XTTS v2... (to może chwilę potrwać)")
    try:
        tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:
        print(f"!!! BŁĄD podczas ładowania modelu: {e}")
        return

    print(f"-> Przetwarzanie próbki głosu z pliku: {PROBKA_WAV}")
    try:
        # Wyciąganie cech charakterystycznych głosu ("latentów") z pliku audio
        gpt_cond_latent, speaker_embedding = tts_api.synthesizer.tts_model.get_conditioning_latents(audio_path=[PROBKA_WAV])
    except Exception as e:
        print(f"!!! BŁĄD podczas przetwarzania pliku audio: {e}")
        return
    
    # Zapisywanie cech głosu do pliku binarnego .pth
    torch.save({
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding
    }, PLIK_WYJSCIOWY_GLOSU)

    print(f"\n[SUKCES] Sklonowany głos został zapisany w pliku: {PLIK_WYJSCIOWY_GLOSU}")
    
    # Automatyczny test zapisanego głosu
    print("\n--- Rozpoczynam automatyczny test zapisanego głosu ---")
    try:
        print(f"-> Generowanie testowego zdania: '{TEKST_DO_TESTOWANIA}'")
        tts_api.tts_to_file(
            text=TEKST_DO_TESTOWANIA,
            language="pl",
            gpt_conditioning_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            file_path=PLIK_WYJSCIOWY_TESTU
        )
        print(f"[SUKCES] Test zakończony! Odsłuchaj plik: {PLIK_WYJSCIOWY_TESTU}")
    except Exception as e:
        print(f"!!! BŁĄD podczas testowego generowania mowy: {e}")


if __name__ == "__main__":
    main()