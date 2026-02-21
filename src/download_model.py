import mlflow
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_best_model(output_path="models/model.pt", metric="accuracy"):
    """
    Downloads the best model from MLflow experiments.
    Assumes the experiment name is 'cats_vs_dogs'.
    """
    experiment_name = "cats_vs_dogs"
    mlflow.set_experiment(experiment_name)
    
    print(f"Searching for best model in experiment '{experiment_name}' based on metric '{metric}'...")
    
    try:
        # Get experiment ID
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        if current_experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            sys.exit(1)
            
        experiment_id = current_experiment.experiment_id
        
        # Search runs
        if metric == "loss":
            order_by = ["metrics.loss ASC"]
        else:
            order_by = ["metrics.accuracy DESC"]

        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=order_by,
            max_results=1
        )
        
        if runs.empty:
            print("No runs found.")
            sys.exit(1)
            
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        print(f"Found best run: {run_id}")
        
        # Download artifact
        client = mlflow.tracking.MlflowClient()
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        local_path = client.download_artifacts(
            run_id,
            "model.pt",
            dst_path=os.path.dirname(output_path)
        )

        # Handle case where MLflow returns directory or file
        if os.path.isdir(local_path):
            downloaded_file = os.path.join(local_path, "model.pt")
        else:
            downloaded_file = local_path

        # Verify model file exists
        if not os.path.exists(downloaded_file):
            raise FileNotFoundError("Downloaded model file not found.")

        # Verify model file is not empty
        if os.path.getsize(downloaded_file) == 0:
            raise ValueError("Downloaded model file is empty.")

        print(f"Model downloaded successfully at {downloaded_file}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_best_model()
