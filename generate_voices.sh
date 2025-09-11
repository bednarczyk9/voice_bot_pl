#!/bin/bash

# --- Konfiguracja ---
# Tutaj możesz zmienić tekst, który ma być przeczytany
TEXT_DO_PRZECZYTANIA="Nazywam się Anna, dzwonię z Powiatowych Klubów Biznesu Trzysta Czternaście. Czy mogę zająć dosłownie minutę, aby przedstawić zaproszenie na lokalne spotkanie dla przedsiębiorców, które odbędzie się w Sandomierzu 10 września?"

# Lista mówców do przetestowania
SPEAKERS=(
    "Claribel Dervla"
    "Daisy Studious"
    "Gracie Wise"
    "Tammie Ema"
    "Alison Dietlinde"
    "Ana Florence"
    "Annmarie Nele"
    "Asya Anara"
    "Brenda Stern"
    "Gitta Nikolina"
    "Henriette Usha"
    "Sofia Hellen"
    "Tammy Grit"
    "Tanja Adelina"
    "Vjollca Johnnie"
    "Nova Hogarth"
    "Maja Ruoho"
    "Uta Obando"
    "Lidiya Szekeres"
    "Chandra MacFarland"
    "Szofi Granger"
    "Camilla Holmström"
    "Lilya Stainthorpe"
    "Zofija Kendrick"
    "Narelle Moon"
    "Barbora MacLean"
    "Alexandra Hisakawa"
    "Alma María"
    "Rosemary Okafor"
    "Ige Behringer"
)

# --- Główna pętla skryptu ---
echo "Rozpoczynam generowanie próbek głosowych..."
echo "Tekst do przeczytania: '$TEXT_DO_PRZECZYTANIA'"
echo "--------------------------------------------------"

for speaker in "${SPEAKERS[@]}"; do
    # Tworzenie nazwy pliku bezpiecznej dla systemu (zamiana spacji na podkreślniki)
    filename=$(echo "$speaker" | tr ' ' '_').wav

    echo "-> Generowanie głosu dla: '$speaker' | Zapisywanie do pliku: $filename"

    # Wywołanie komendy TTS
    tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \
        --text "$TEXT_DO_PRZECZYTANIA" \
        --speaker_idx "$speaker" \
        --language_idx "pl" \
        --out_path "$filename"

    # Sprawdzenie, czy komenda się powiodła
    if [ $? -eq 0 ]; then
        echo "   ...Sukces!"
    else
        echo "   ...BŁĄD! Nie udało się wygenerować głosu dla '$speaker'."
    fi
    echo "--------------------------------------------------"
done

echo "Zakończono. Wszystkie pliki zostały zapisane w bieżącym folderze."
