# Topic Classification Service

An API service for academic topic classification based on [OpenAlex's](https://github.com/ourresearch/openalex-topic-classification/tree/main) predictor model.

## Prerequisites

- Python 3.10+
- curl or wget for downloading model artifacts
  
## Model and Artifacts

1. Download the trained model and artifacts:

```bash
wget https://zenodo.org/records/10568402/files/topic_classifier_v1_artifacts.tar.gz
```

2. Create models directory and extract artifacts:

```bash
mkdir -p model
tar -xzf topic_classifier_v1_artifacts.tar.gz -C model
```

## Development Setup

1. Install the uv package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt --no-cache-dir
```

4. Start the development server:

```bash
uvicorn main:app --reload --port <PORT>
```

## API ENDPOINTS

### Health Check

- **Endpoint:** `/health_check`
- **Method:** `GET`
- **Description:** Checks the service and model health.
- **Example Request:**

  ```bash
  curl http://localhost:<PORT>/health_check
  ```

- **Example Response:**

  ```json
  {
    "status": "healthy",
    "model": "loaded"
  }
  ```

### Single Paper Prediction

- **Endpoint:** `/single`
- **Method:** `POST`
- **Description:** Predicts topics for a single academic paper.
- **Input Data Format:**

  ```json
  [
    {
      "title": "Multiplication of matrices of arbitrary shape on a data parallel computer",
      "abstract_inverted_index": {
        "Some": [0],
        "level-2": [1],
        "and": [2],
        "level-3": [3],
        "Distributed": [4],
        "Basic": [5],
        "Linear": [6],
        "Algebra": [7],
        "Subroutines": [8],
        "(DBLAS)": [9],
        "that": [10],
        "have": [11],
        "been": [12],
        "implemented": [13],
        "on": [14, 26],
        "the": [15, 27],
        "Connection": [16],
        "Machine": [17],
        "system": [18],
        "CM-200": [19],
        "are": [20],
        "described.": [21],
        "No": [22],
        "assumption": [23],
        "is": [24],
        "made": [25],
        "shape": [28],
        "or": [29],
        "...": [30]
      },
      "journal_display_name": "Fire Safety Science",
      "referenced_works": [
        "https://openalex.org/W183327403",
        "https://openalex.org/W1851212222",
        "https://openalex.org/W1967958850",
        "https://openalex.org/W1988425770",
        "https://openalex.org/W1991286031",
        "https://openalex.org/W2029342163",
        "https://openalex.org/W2045381439",
        "https://openalex.org/W2053280233",
        "https://openalex.org/W2071782145",
        "https://openalex.org/W2083202979",
        "https://openalex.org/W2104487100",
        "https://openalex.org/W4234919994"
      ],
      "inverted": true
    }
  ]
- **Example Request:**

  ```python
  import requests
  import json

  url = "http://localhost:<PORT>/single"
  headers = {"Content-Type": "application/json"}
  with open("test_samples/test_json_single.json", "r") as f:
      data = json.load(f)
  response = requests.post(url, headers=headers, json=data)
  print(response.json()) 
  ```

- **Example Response:**

    ```json
    [[{"topic_id": 10829, "topic_label": "829: Networks on Chip in System-on-Chip Design", "topic_score": 0.9978}, {"topic_id": 10054, "topic_label": "54: Parallel Computing and Performance Optimization", "topic_score": 0.9962}, {"topic_id": 11522, "topic_label": "1522: Design and Optimization of Field-Programmable Gate Arrays and Application-Specific Integrated Circuits", "topic_score": 0.9909}]]
    ```

### Batch Paper Prediction

- **Endpoint:** `/batch`
- **Method:** `POST`
- **Description:** Predicts topics for a batch of academic papers.
- **Example Request:**

  ```python
  import requests
  import json

  url = "http://localhost:<PORT>/batch"
  headers = {"Content-Type": "application/json"}
  with open("test_samples/test_json_batch.json", "r") as f:
      data = json.load(f)
  response = requests.post(url, headers=headers, json=data)
  print(response.json()) 
  ```

- **Example Response:**

  ```json
  [[{"topic_id": 10829, "topic_label": "829: Networks on Chip in System-on-Chip Design", "topic_score": 0.9978}, {"topic_id": 10054, "topic_label": "54: Parallel Computing and Performance Optimization", "topic_score": 0.9962}, {"topic_id": 11522, "topic_label": "1522: Design and Optimization of Field-Programmable Gate Arrays and Application-Specific Integrated Circuits", "topic_score": 0.9909}], [{"topic_id": 10110, "topic_label": "110: Seismicity and Tectonic Plate Interactions", "topic_score": 0.9995}, {"topic_id": 12157, "topic_label": "2157: Machine Learning for Mineral Prospectivity Mapping", "topic_score": 0.9933}, {"topic_id": 10399, "topic_label": "399: Characterization of Shale Gas Pore Structure", "topic_score": 0.991}]]
  ```

## License

This project uses OpenAlex's topic classification model. Please refer to their [license](https://github.com/ourresearch/openalex-topic-classification/blob/main/LICENSE) for terms of use.
