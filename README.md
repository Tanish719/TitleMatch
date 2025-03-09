
# **Title Similarity API**  

This FastAPI-based project provides an endpoint to find the most similar title from a given list using NLP embeddings. It leverages the **all-MiniLM-L6-v2** model from Hugging Face and computes similarity using **cosine similarity**. The API is built with **FastAPI** and standard Python dependencies for efficiency and ease of use.

  

## Technologies used

1.  **[FastAPI](https://fastapi.tiangolo.com)**: A modern, fast, web framework for building APIs with python. The reason FastAPI was chosen for this model is due to its async support which is useful for handling multiple requests for NLP models. It is much more user-friendly to python users than other API providers. 

2. **[Sentence-Transformer](https://sbert.net/)**: To encode sentences and compute embeddings along with cosine similarity from the [Scikit](https://scikit-learn.org/stable/) library to evaluate similarity strength.

3. **[Numpy](https://numpy.org/)**: For handling numerical operations


## Cosine similarity vs L2 (Euclidean) similarity
When comparing two vectors (e.g., sentence embeddings in NLP), we can use different **distance metrics** to measure their similarity. The two most common ones are:

 1. Cosine similarity: Measures angles between two vectors
	 $$ cos\theta =\frac{ \sum_i A_iB_i}{(\sum_i A_i^2)(\sum_i B_i^2)}  $$
	 The result is always between -1 (least related) to 1 (perfect match). 
	 ```python
	 from sklearn.metrics.pairwise import cosine_similarity
	import numpy as np

	A = np.array([1, 2, 3]).reshape(1, -1)
	B = np.array([4, 5, 6]).reshape(1, -1)

	similarity = cosine_similarity(A, B)
	print(similarity)  # Closer to 1 means more similar
	
	output: 0.97463185
	```

 2. L2 similarity: Measures absolute distance between two vectors
	 $$   ||A-B||_2 = \sqrt{\sum_i(A_i-B_i)^2}  $$
	The smaller the euclidean distance is the more similar the two vectors 		are to one another.
	 ```python
	 from sklearn.metrics.pairwise import euclidean_distanced
	import numpy as np

	A = np.array([1, 2, 3]).reshape(1, -1)
	B = np.array([4, 5, 6]).reshape(1, -1)

	similarity = cosine_similarity(A, B)
	print(similarity)  # Closer to 1 means more similar
	
	output: 5.19615242
	```

It is important to note that the euclidean distance is **sensitive** to the magnitude of the vectors and would hence, be more useful for numerical analysis rather than analyzing NLP embeddings which vary in scale. As angles does not rely on the magnitude of the vectors rather its direction which is -   best for text/NLP tasks because sentence embeddings tend to have different lengths. Hence, **cosine similarity is the best choice** for this particular project. 

## Installation instructions (Terminal)
#### Step 1: Clone the repository
```bash
 git clone https://github.com/Tanish719/TitleMatch.git
```

#### Step 2: Install dependencies
```bash
pip3 install -r requirements.txt
```

#### Step 3: Running the API locally 
```bash
uvicorn Main:app
```
If the file runs with no problems the terminal will display
```bash
INFO: Started server process [51360]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
Copy the http link and paste it in your browser and the following message should appear.
```
{"message":"Welcome to the Sentence Similarity API"}
```
## API Usage

This endpoint accepts two lists of sentences: reference sentences and target sentences. It returns the most similar target sentence for each reference sentence based on cosine similarity.
#### Request body (input.json file)
```json
{
  "reference_sentences": [
    "Higgs boson in particle physics",
    "other"
  ],
  "target_sentences": [
    "Best soup recipies",
    "Particle physics at CERN",
    "Basel activites"
  ]
}
```
#### Running the request
Make sure the API is active and run this command in another terminal window
```bash
python3 send_request.py
```
#### Response body
```bash
{
  "Top_result": "Particle physics at CERN.", "Similarity_score": "0.5"
}
```
