from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import DeepProtMultiKernelCNN


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "deepprot_multikernel_weighted_best.pth"
VOCAB_PATH = PROJECT_DIR / "data" / "processed" / "go_vocabulary_aligned.csv"

MAX_LENGTH = 3000
THRESHOLD = 0.46

AMINO_ACID_MAP = {
    aa: i + 1
    for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")
}


class DeepProtPredictor:

    def __init__(self):

        self.device = torch.device("cpu")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"GO vocabulary not found: {VOCAB_PATH}")

        self.go_vocab = pd.read_csv(VOCAB_PATH)

        required_columns = [
            "GO_ID",
            "GO_Name",
            "GO_Namespace"
        ]

        for column in required_columns:
            if column not in self.go_vocab.columns:
                raise ValueError(f"Missing vocabulary column: {column}")

        if len(self.go_vocab) != 226:
            raise ValueError("Expected exactly 226 GO terms.")

        self.model = DeepProtMultiKernelCNN(
            vocab_size=21,
            embedding_dim=32,
            num_filters=32,
            num_classes=226,
            dropout=0.4
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sequence, top_k=20):

        sequence = "".join(str(sequence).upper().split())

        if not sequence:
            raise ValueError("Protein sequence cannot be empty.")

        invalid = sorted(set(sequence) - set(AMINO_ACID_MAP))

        if invalid:
            raise ValueError(
                f"Unsupported amino acid characters: {invalid}"
            )

        original_length = len(sequence)
        truncated = original_length > MAX_LENGTH

        sequence = sequence[:MAX_LENGTH]

        encoded = [
            AMINO_ACID_MAP[aa]
            for aa in sequence
        ]

        input_tensor = torch.zeros(
            (1, MAX_LENGTH),
            dtype=torch.long
        )

        input_tensor[0, :len(encoded)] = torch.tensor(
            encoded,
            dtype=torch.long
        )

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        results = self.go_vocab[
            ["GO_ID", "GO_Name", "GO_Namespace"]
        ].copy()

        results["Confidence"] = probabilities
        results["Confidence_Percent"] = (
            probabilities * 100
        ).round(2)

        results["Predicted"] = (
            probabilities >= THRESHOLD
        )

        results = results.sort_values(
            "Confidence",
            ascending=False
        ).reset_index(drop=True)

        return {
            "sequence_length": original_length,
            "processed_length": len(sequence),
            "truncated": truncated,
            "threshold": THRESHOLD,
            "total_predicted_terms": int(
                results["Predicted"].sum()
            ),
            "predictions": results.head(top_k),
            "all_predictions": results
        }
