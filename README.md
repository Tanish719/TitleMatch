


# TitleMatch

Finds the most similar title from an inputed list from a JSON file.

  

## Technologies used

1.  **[FastAPI](https://fastapi.tiangolo.com)**: A modern, fast, web framework for building APIs with python. 

2. **[Sentence-Transformer](https://sbert.net/)**: To encode sentences and compute embeddings along with cosine similarity from the [Scikit](https://scikit-learn.org/stable/) library to evaluate similarity strength.

4. **[Numpy](https://numpy.org/)**: For handling numerical operations

## Installation instructions
#### Step 1: Clone the repository
```bash
 git clone https://github.com/Tanish719/TitleMatch
```

#### Step 2: Install dependencies
```bash
pip3 install -r requirements.txt
```

#### Step 3: Running the API locally 
```bash
uvicorn main:app
```
If the file has run with no problems the terminal will display
```bash
INFO: Started server process [51360]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
Copy the http link and paste it in your browser and the following message should be visible.
```
{"message":"Welcome to the Sentence Similarity API"}
```
## API Usage

This endpoint accepts two lists of sentences: reference sentences and target sentences. It returns the most similar target sentence for each reference sentence based on cosine similarity.
#### Request body (input.json file)
```json
{
  "reference_sentences": [
    "This is a reference sentence.",
    "Another reference sentence."
  ],
  "target_sentences": [
    "This is the target sentence.",
    "Yet another target sentence."
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
  "Top_result": "This is the target sentence."
}
```
