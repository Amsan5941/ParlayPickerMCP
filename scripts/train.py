"""Entry point: train all models."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import train_all_models, train_single_model

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        model_name = sys.argv[1]
        print(f"Training single model: {model_name}")
        result = train_single_model(model_name)
        print(f"Result: {result}")
    else:
        print("Training all models...")
        results = train_all_models()
        print("\n=== Training Summary ===")
        for name, metrics in results.items():
            print(f"  {name}: {metrics}")
        print("Done!")
