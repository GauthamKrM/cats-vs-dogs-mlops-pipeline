import requests
import argparse
import sys
import os

def smoke_test(url, image_path=None):
    """
    Validates that the service is running and (optionally) can make predictions.
    Exits with status 1 if any check fails.
    """
    print(f"Running smoke test against {url}...")
    
    # 1. Health Verification
    try:
        health_url = f"{url.rstrip('/')}/health"
        print(f"Checking {health_url}...")
        resp = requests.get(health_url, timeout=5)
        
        if resp.status_code == 200:
            print(f"Health check PASSED: {resp.json()}")
        else:
            print(f"Health check FAILED: Status {resp.status_code} - {resp.text}")
            sys.exit(1)
            
        # Optional: Fail if health status is explicitly 'unhealthy'
        if resp.json().get("status") == "unhealthy":
            print("Service is responding but reports 'unhealthy' (Model not loaded?)")
            sys.exit(1) 
    except Exception as e:
        print(f"Health check ERROR: {e}")
        sys.exit(1)

    # 2. Prediction Verification
    if image_path:
        predict_url = f"{url.rstrip('/')}/predict"
        print(f"Testing prediction at {predict_url} with {image_path}...")
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            sys.exit(1)

        try:
            with open(image_path, "rb") as f:
                files = {"file": f}
                resp = requests.post(predict_url, files=files, timeout=10)
                
            if resp.status_code == 200:
                result = resp.json()
                print(f"Prediction PASSED: {result}")
                if "prediction" not in result:
                    print(f"Invalid response format: {result}")
                    sys.exit(1)
            else:
                print(f"Prediction FAILED: Status {resp.status_code} - {resp.text}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Prediction ERROR: {e}")
            sys.exit(1)
    else:
        print("Skipping prediction test (no image provided)")

    print("All smoke tests passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke tests against the deployed inference service.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--image", help="Path to an image file for prediction testing")
    args = parser.parse_args()
    
    smoke_test(args.url, args.image)
