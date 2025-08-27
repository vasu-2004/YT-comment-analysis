# register_model.py
import json
import mlflow
import logging
import os

# --- Logging configuration ---
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading model info: %s', e)
        raise


def register_model(model_info: dict):
    """Register the model to the MLflow Model Registry (local)."""
    try:
        # Always use local tracking server
        tracking_uri = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[register_model] Using MLflow at: {tracking_uri}")

        # Build model URI from run + artifact path
        model_uri = f"runs:/{model_info['run_id']}/{model_info['artifact_path']}"

        # Use registered name if available, else fallback
        model_name = model_info.get("registered_model_name", "yt_chrome_plugin_model")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Move the model to Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug("Model %s v%s registered and moved to Staging.", model_name, model_version.version)
        print(f"✅ Registered {model_name} v{model_version.version} → Staging")

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        register_model(model_info)
    except Exception as e:
        logger.error('Failed to complete registration: %s', e)
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
