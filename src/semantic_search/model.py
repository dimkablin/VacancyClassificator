"""implementation semantic search model from HF"""
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class SemanticClassificator:
    """Semantic search classificator"""

    def __init__(self) -> None:
        self.classes = None
        self.classes_embeddings = None
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

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

        self.classes_embeddings = self.get_embeddings(self.classes)

    def get_embeddings(self, texts: list[str]):
        """return embedding of the model"""
        encoded_text = self.tokenizer(texts, padding=True, truncation=True,
                                      max_length=64, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_text)

        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)

        return embeddings

    def predict(self, text: str, thresh=None):
        """Find the nearest sentance

        Args:
            text (str): query or input sentance
            treshold (float, optional): Treshold. If None (Default) return all classes.

        Returns:
           dict: keys: "class", "similarity" 
        """

        text_embeddings = self.get_embeddings([text])
        similarities = cosine_similarity(text_embeddings, self.classes_embeddings)[0]

        result = []

        for class_, similarity in zip(self.classes, similarities):
            result.append({"class": class_, "similarity": similarity})

        result = sorted(result, key=lambda x: x["similarity"], reverse=True)

        if thresh is not None:
            result = [res for res in result if res["similarity"] > thresh]

        return result
