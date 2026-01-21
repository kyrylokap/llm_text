import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


def test_check_configuration():
    print("\n--- 1. CONFIGURATION CHECK ---")
    assert API_KEY,"API Key: NOT FOUND! Check your .env file"
    assert MODEL_NAME , "Model Name: NOT FOUND! Check your .env file"



def test_run_connection_test():
    print(f"\n--- 2. MODEL CONNECTION TEST:  ---")

    try:
        client = Groq(api_key=API_KEY)

        print("Sending simple request...")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Write one sentence: 'Connection is working correctly'."
                }
            ],
            model=MODEL_NAME,
        )

        response_text = chat_completion.choices[0].message.content
        assert response_text is not None and len(response_text) > 0, "Response ERROR!"

        print("\nSUCCESS! Response received.")
        print(f"AI Response: \"{response_text}\"")

    except Exception as e:
        print(f"\nCRITICAL ERROR:")
        print(f"{str(e)}")
