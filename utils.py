from langchain_core.embeddings import Embeddings
from google import genai

class CustomGeminiEmbeddings(Embeddings):
    """
    Custom wrapper to embed documents one by one.
    This bypasses the issue in the python SDK where batching multiple documents
    with varying lengths results in a mismatched number of returned embeddings.
    """
    def __init__(self, model_name="gemini-embedding-2"):
        self.client = genai.Client()
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name, 
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
