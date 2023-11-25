"""implementation semantic search model from HF"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticClassificator:
    """Semantic search classificator"""

    def __init__(self) -> None:
        self.classes = None
        self.classes_embeddings = None
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )


    def init_classes(self, classes):
        """Set possible classes. 

        Args:
            classes (list[str] or str): list of possible classes OR path to the .txt file

        # Example 1:
            ```python
            model = Classificator(device="cuda")
            model.init_classes("src/zero_shot_classification/classes.txt")
            ```

        # Example 2:
            ```python
            model = Classificator(device="cuda")
            model.init_classes(['class1', 'class2'])
            ```
        """
        # Clear classes
        self.classes = []

        if isinstance(classes, list):
            self.classes = classes
        else:
            with open(classes, "r", encoding="utf-8") as file:
                for line in file:
                    self.classes.append(line.strip())

        self.classes_embeddings = self.model.encode(self.classes)

    def predict(self, text: str, thresh=None):
        """Find the nearest sentance

        Args:
            text (str): query or input sentance
            treshold (float, optional): Treshold. If None (Default) return all classes.

        Returns:
           dict: keys: "class", "similarity" 
        """

        text_embeddings = self.model.encode([text])
        similarities = cosine_similarity(text_embeddings, self.classes_embeddings)[0]

        result = []

        for class_, similarity in zip(self.classes, similarities):
            result.append({"class": class_, "similarity": similarity})

        result = sorted(result, key=lambda x: x["similarity"], reverse=True)

        if thresh is not None:
            result = [res for res in result if res["similarity"] > thresh]

        return result
