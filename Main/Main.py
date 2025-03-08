from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


class SimilarityRequest (BaseModel):
    reference_sentences: List[str]
    target_sentences: List[str]

class TopResultResponse(BaseModel):
    Top_result: str


@app.post("/find_best_match", response_model=TopResultResponse)
async def find_best_match(request: SimilarityRequest):
 
 
    reference_embeddings = model.encode(request.reference_sentences)  # Multiple references
    sentence_embeddings = model.encode(request.target_sentences)


    best_reference_sentence = None
    best_similarity_score = -1  # Initialize with a very low score

# Compare each reference sentence with the list of sentences
    for reference_embedding, reference_sentence in zip(reference_embeddings, request.reference_sentences):
    # Compute cosine similarity between this reference and all sentences in the list
        similarities = cosine_similarity([reference_embedding], sentence_embeddings)[0]  # 1D array of similarities
    
    # Find the most similar sentence to this reference sentence
        most_similar_index = np.argmax(similarities)
        most_similar_score = similarities[most_similar_index]
    
    # If this reference sentence's best similarity score is the highest so far, update
        if most_similar_score > best_similarity_score:
            best_similarity_score = most_similar_score
            best_reference_sentence = request.target_sentences[most_similar_index]
    
    return TopResultResponse(Top_result=best_reference_sentence)


@app.get("/")
async def root():
    return {"message": "Welcome to the Sentence Similarity API"}


