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
   
    Similartiy_score: float


@app.post("/find_best_match", response_model=TopResultResponse)
async def find_best_match(request: SimilarityRequest):
 
 
    reference_embeddings = model.encode(request.reference_sentences)  
    sentence_embeddings = model.encode(request.target_sentences)


    best_target_sentence = None
    best_reference_sentence = None
    best_similarity_score = -1  


    for reference_embedding, reference_sentence in zip(reference_embeddings, request.reference_sentences):
    
        similarities = cosine_similarity([reference_embedding], sentence_embeddings)[0]  
    
    
        most_similar_index = np.argmax(similarities)
        most_similar_score = similarities[most_similar_index]
    
    
        if most_similar_score > best_similarity_score:
            best_similarity_score = most_similar_score
            best_target_sentence = request.target_sentences[most_similar_index]
            best_reference_sentence = reference_sentence
    
    return TopResultResponse(Top_result=best_target_sentence, Similartiy_score=round(best_similarity_score, 3))


@app.get("/")
async def root():
    return {"message": "Welcome to the Sentence Similarity API"}


