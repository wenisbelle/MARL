import mlflow
import argparse
from hyperopt import hp, fmin, tpe
from data_harvesting.train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=False, help="MLflow experiment ID", dest="experiment_name")
argparse.add_argument("-R", type=str, required=False, help="MLflow run ID", dest="run_id")
args = argparse.parse_args()

mlflow.set_tracking_uri("file:./mlruns")

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_name if args.experiment_name else "default")

    train(config, run_name=args.run_id)