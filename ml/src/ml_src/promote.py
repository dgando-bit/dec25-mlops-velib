from mlflow import MlflowClient
from shared.config import settings

def promote(version: int, alias: str = "production") -> None:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    client.set_registered_model_alias(
        name=settings.registered_model_name,
        alias=alias,
        version=str(version),
    )
    print(f"Version {version} promue avec alias '{alias}'.")

if __name__ == '__main__':
    import sys
    version = int(sys.argv[1])
    alias   = sys.argv[2] if len(sys.argv) > 2 else "production"
    promote(version, alias)