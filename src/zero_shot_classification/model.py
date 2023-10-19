"""Model initialization."""
from transformers import pipeline


class Classificator:
    """Zero Shot Classificator on the given classes."""
    def __init__(self, device='cpu'):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="cointegrated/rubert-base-cased-nli-threeway",
            device=device
        )

        self.classes = None

    def init_classes(self, path):
        """Set possible classes."""
        # Clear classes
        self.classes = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                self.classes.append(line.strip())

    def predict(self, text, allow_multi_labels=True, thresh=0.5):
        """Predict classes function."""
        assert self.classes is not None, "Classes cannot be empty."

        # Use pipeline
        result = self.pipe(text, self.classes, allow_multi_labels=allow_multi_labels)

        # Filter labels by its score
        result = [
            predict for predict, score in zip(result['labels'], result['scores']) if score > thresh
        ]

        return result
