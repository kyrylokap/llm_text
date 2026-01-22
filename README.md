# Groq Hosted Model API

A FastAPI application using Groq hosted model Llama.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/way/to/repository
```

### 2. Create and activate a virtual environment

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate

```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install requirements

```bash
pip install -r requirements.txt

```

### 4. Configuration

Create a `.env` file in the root directory and add your Groq API key:

```ini
# Get your API Key here: https://console.groq.com/keys
GROQ_API_KEY=gsk_YOUR_API_KEY_HERE
MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct

```

---

## 🚀 Running the Application

Start the server using Uvicorn with hot-reload enabled:

```bash
uvicorn app.main:app --reload

```

The server will start at `http://127.0.0.1:8000`.

---

## 🧪 Testing

### cURL Example

To test via terminal, use the command below.

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -F "message=Does this look like strep throat?" \
     -F "history=[]" \
     -F "images=@/home/vasyl/throat.jpeg" \
     -F "use_functions=true" \
     -F "mode=api" \
     -F "k=3"
```
