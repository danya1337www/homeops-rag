import os 

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class GeminiEmbedder: 
    def __init__(self) -> None: 
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key: 
            raise ValueError(
                "Нет GEMINI_API_KEY"
            )

        self.model = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-2"
        )

        self.client = genai.Client(api_key = api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray: 
        prepared_texts = [
            f"title: none | text: {text}"
            for text in texts
        ]

        contents = [ 
            types.Content(parts = [types.Part.from_text(text = text)])

            for text in prepared_texts
        ]

        response = self.client.models.embed_content(
            model = self.model,
            contents = contents,
            config = types.EmbedContentConfig(output_dimensionality=768),
        )

        vectors = [
            embedding.values 
            for embedding in response.embeddings
        ]

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        prepared_query = (f"task: question answering | query {query}")

        response = self.client.models.embed_content(
            model = self.model,
            contents = prepared_query,
            config = types.EmbedContentConfig(output_dimensionality = 768)
        )

        return np.asarray(response.embeddings[0].values, dtype = np.float32)
