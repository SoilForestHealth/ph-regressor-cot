def create_batch(request_id: str, prompt: str, temperature: float = 0.0) -> list:

    return {
        "custom_id": request_id, 
        "request": {
            "anthropic_version": "vertex-2023-10-16", 
            "messages": [
                {
               "role": "user", 
               "content": [
                  {"type": "text", 
                   "text": prompt
                   }]
                }
            ], 
            "max_tokens": 4096, 
            "temperature": temperature}}
