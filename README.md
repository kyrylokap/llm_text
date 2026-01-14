# Groq Hosted Model API

## Instalacja
1. Sklonuj repo:
   git clone https://github.com/kyrylokap/llm_text
2. Utwórz i aktywuj virtualenv:
   python -m venv .venv
   source .venv/bin/activate
   .venv\Scripts\activate
3. Zainstaluj wymagania:
   pip install -r requirements.txt
4. Utwórz plik `.env` z kluczem API:
   GROQ_API_KEY=twoj_klucz

## Uruchomienie
uvicorn app.main:app --reload 

## Testowanie
curl -X POST http://127.0.0.1:8000/docs \
-H "Content-Type: application/json" \
-d '{"symptoms":"fever and cough", "k":3, "mode":"api", "use_functions":false}'
