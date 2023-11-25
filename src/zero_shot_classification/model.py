"""Model initialization."""
from transformers import pipeline


class ZeroShotClassificator:
    """Zero Shot Classificator on the given classes."""
    def __init__(self, device='cpu'):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=device
        )

        self.classes = None

    def init_classes(self, classes):
        """Set possible classes."""
        # Clear classes
        self.classes = []

        if isinstance(classes, list):
            self.classes = classes
        else:
            with open(classes, "r", encoding="utf-8") as file:
                for line in file:
                    self.classes.append(line.strip())

    def predict(self, text, allow_multi_labels=True, thresh=0.5):
        """Predict classes function."""
        assert self.classes is not None, "Classes cannot be empty."

        # Use pipeline
        preds = self.pipe(text, self.classes, allow_multi_labels=allow_multi_labels)

        # Filter labels by its score
        result = []
        if allow_multi_labels:
            result = [
                pred for pred, score in zip(preds['labels'], preds['scores']) if score > thresh
            ]
        elif not result:
            index = preds['scores'].index(max(preds['scores']))
            result.append(preds['labels'][index])

        return result
