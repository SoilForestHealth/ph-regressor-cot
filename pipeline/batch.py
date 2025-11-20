def create_batch(request_id: str, prompt: str, temperature: float = 0.0) -> list:

    return {
        "key": request_id,
        "request": {
            "contents": [
                {
                    "role": "user", 
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature
            }
        }
    }
