
# Full CI/CD Pipeline – ML Engineering for Production (Assignment 5)

This project implements a full CI/CD pipeline for a simple machine learning model using GitHub Actions, MLflow, DVC, and Docker. The pipeline trains a classifier, logs metrics to MLflow, validates that accuracy is above a threshold, and only then runs a mock Docker deployment step.

---

## Project structure

- `train.py` – trains a classifier, logs accuracy to MLflow, and writes the current Run ID to `model_info.txt`.
- `check_threshold.py` – reads the Run ID from `model_info.txt`, fetches the run from MLflow, and fails if accuracy < 0.85.
- `.github/workflows/pipeline.yml` – GitHub Actions workflow with two jobs: `validate` and `deploy`.
- `Dockerfile` – simple image that installs dependencies and simulates downloading a model using `ARG RUN_ID`.
- `requirements.txt` – Python dependencies.
- `.dvc/`, `.dvcignore` – DVC configuration for data versioning.

---

## How the pipeline works

### 1. Validate job

Triggered on pushes and pull requests to `main`:

1. Checkout the repository.
2. Set up Python 3.10 and install dependencies from `requirements.txt`.
3. (Optional) Run `dvc pull` to download the dataset tracked by DVC.
4. Run `python train.py`:
   - Trains the model.
   - Logs accuracy and other info to MLflow.
   - Saves the MLflow Run ID to `model_info.txt`.
5. Upload `model_info.txt` (and `mlflow.db`) as artifacts for the next job.

If training or artifact upload fails, the pipeline stops here.

### 2. Deploy job

This job depends on `validate`:

1. Checkout the repository and set up Python.
2. Install dependencies.
3. Download the `model-info` and `mlflow-db` artifacts.
4. Run `python check_threshold.py`:
   - Reads the Run ID from `model_info.txt`.
   - Loads the corresponding run from MLflow.
   - Reads the logged accuracy.
   - Exits with error if accuracy < 0.85.
5. If the check passes, run a mock Docker build:

```bash
echo "Building Docker image for Run ID: $(cat model_info.txt)"
```

If the accuracy is below 0.85, this step is skipped and the `deploy` job fails.

---

## Dockerfile

The `Dockerfile`:

- Uses `python:3.10-slim` as the base image.
- Accepts an `ARG RUN_ID`.
- Installs dependencies from `requirements.txt`.
- Copies the project code into `/app`.
- Simulates downloading the model:

```dockerfile
RUN echo "Downloading model for Run ID: ${RUN_ID}"
```

- Uses `CMD ["python", "train.py"]` as the default command.

This is a mock deployment to demonstrate how the model could be packaged once it passes validation.

---

## Running locally

1. Create and activate a virtual environment (optional).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (If using DVC data) pull the dataset:

```bash
dvc pull
```

4. Run training:

```bash
python train.py
```

5. Check the threshold manually:

```bash
python check_threshold.py
```

---

## Evidence for the assignment

- **Failed run**: A GitHub Actions run where `deploy` fails at `Check accuracy threshold` because accuracy < 0.85.
- **Successful run**: A GitHub Actions run where `deploy` completes and the `Build Docker image` step runs, echoing the Run ID.

These screenshots demonstrate that the CI/CD pipeline gates deployment based on model performance.
```
![Failed run](image.png)
![Successful run](image-1.png)
```
---

## Project structure

```text
.
├── .dvc/                 # DVC configuration
├── .github/workflows/
│   └── pipeline.yml      # CI/CD pipeline (GitHub Actions)
├── data/                 # Dataset (tracked with DVC)
├── mlruns/               # MLflow experiment tracking
├── tests/                # Unit tests
├── .dvcignore
├── .gitignore
├── check_threshold.py    # Model validation logic
├── Dockerfile            # Container setup
├── mlflow.db             # MLflow database
├── model_info.txt        # Saved model metadata
├── requirements.txt      # Dependencies
├── train.py              # Training script
└── README.md
