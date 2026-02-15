# Cats vs Dogs MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs), featuring experiment tracking, model packaging, Docker containerization, and a CI/CD workflow.

## Project Structure

```
├── .github/workflows/   # CI/CD pipelines (GitHub Actions)
├── data/                # Dataset (managed by DVC, not in git)
├── deploy/              # Kubernetes manifests
├── models/              # Trained models (managed by DVC/MLflow)
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── app.py           # FastAPI inference service
│   ├── data_loader.py   # Data loading and preprocessing logic
│   ├── download_model.py# Script to download model from MLflow
│   ├── evaluate.py      # Model evaluation script
│   └── train.py         # Training script with MLflow tracking
├── tests/               # Unit and integration tests
├── .dockerignore        # Docker build context exclusions
├── .gitignore           # Git exclusions
├── docker-compose.yml   # (Optional) Local composition
├── Dockerfile           # Container definition
├── dvc.yaml             # DVC pipeline stages
├── params.yaml          # DVC/Training hyperparameters
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GauthamKrM/cats-vs-dogs-mlops-pipeline
    cd cats-vs-dogs-mlops-pipeline
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory (copy from `.env.example` if available) to store secrets like MLflow credentials (if using a remote server like Dagshub).
    ```ini
    MLFLOW_TRACKING_URI=https://dagshub.com/...
    MLFLOW_TRACKING_USERNAME=...
    MLFLOW_TRACKING_PASSWORD=...
    ```

## Model Development (DVC & MLflow)

We use **DVC** to manage the training pipeline and **MLflow** for experiment tracking.

### 1. Prepare Data
Ensure your data is in the `data/` directory, organized as:
```
data/
  Cat/
  Dog/
```

### 2. Run Pipeline (Experiment Repro)
To reproduce the entire pipeline (prepare -> train -> evaluate):
```bash
dvc exp run
```
This command will:
-   Execute `src/train.py` using parameters from `params.yaml`.
-   Log metrics, params, and the model artifact to MLflow.
-   Execute `src/evaluate.py` to generate `metrics.json`.

**View Metrics & Diffs:**
```bash
dvc metrics show
dvc metrics diff  # Compare with previous run
```

### 3. Hyperparameters
Modify `params.yaml` to change training configurations (epochs, batch_size, learning_rate).
Running `dvc exp run` again will detect changes and re-run necessary stages.

## Docker & Inference

The inference service is built with **FastAPI**.

### Build Docker Image
```bash
docker build -t cats-vs-dogs-inference .
```

### Run Locally
```bash
docker run -p 8000:8000 --env-file .env cats-vs-dogs-inference
```
Access the API documentation at `http://localhost:8000/docs`.

### API Endpoints
-   **GET /health**: Check service status.
-   **POST /predict**: Upload an image to get a classification (Cat/Dog).

## Testing

Run unit tests ensuring code quality and data integrity:
```bash
pytest tests/
```

## CI/CD Pipeline

The project includes a **GitHub Actions** workflow (`.github/workflows/ci.yml`) that runs on every push to `main`.

**CI Steps:**
1.  **Checkout Code**
2.  **Install Python Dependencies**
3.  **Run Unit Tests** (`pytest`)
4.  **Download Model**: Fetches the best trained model from MLflow.
5.  **Build Docker Image**: Packages the application with the model.
6.  **Push to Registry**: Pushes the image to Docker Hub / GHCR.

**CD Steps:**
-   Updates the Kubernetes manifests in `deploy/` with the new image tag (GitOps approach).
-   (Optional) Triggers ArgoCD or applies manifests to a cluster.

## Deployment (Minikube)

### Prerequisites
-   Minikube installed & running (`minikube start`)
-   `kubectl` installed

### 1. Build Image in Minikube's Docker Engine
To ensure Minikube can find your image without a registry push:
```bash
minikube docker-env | Invoke-Expression  # Windows PowerShell
# OR
eval $(minikube docker-env)              # Linux/Mac
```
Now build the image:
```bash
docker build -t cats-vs-dogs-inference .
```

### 2. Apply Manifests
Deploy the application and service to the cluster:
```bash
kubectl apply -f deploy/deployment.yaml
kubectl apply -f deploy/service.yaml
```

### 3. Verification
Check the status of your pods:
```bash
kubectl get pods
```

### 4. Access the Service
Since we use `NodePort` or `ClusterIP`, port-forward to access locally:
```bash
kubectl port-forward svc/cats-vs-dogs-service 8000:80
```
Now access the API at `http://localhost:8000/docs`.

### 5. Smoke Test
Run the post-deployment smoke test:
```bash
python tests/smoke_test.py --url http://localhost:8000
```
## Monitoring

-   **FastAPI Instrumentation**: The app exposes Prometheus metrics at `/metrics` (enabled via `prometheus-fastapi-instrumentator`).
-   **MLflow**: Track model performance over time.
