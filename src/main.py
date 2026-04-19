import argparse
from src.train_classifier import main as train_classifier
from src.train_anomaly import main as train_anomaly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bearing Fault Detection System")
    parser.add_argument("--mode", type=str, choices=["classifier", "anomaly"], default="classifier", help="Choose model to train")
    args = parser.parse_args()

    if args.mode == "classifier":
        print("Starting Supervised Classifier Training...")
        train_classifier()
    elif args.mode == "anomaly":
        print("Starting Unsupervised Anomaly Detector Training...")
        train_anomaly()
        