from abc import ABC, abstractmethod
import numpy as np
import torch
from torchvision import transforms
import sys
from PIL import Image


class DigitClassificationInterface(ABC):
    # interface for all classification classes
    @abstractmethod
    def predict(self, data):
        pass


class CNN(DigitClassificationInterface):
    def __init__(self, model=None):
        self.model = model

    def predict(self, data: torch.Tensor) -> int:
        print(f"\nThis is CNNModel. Input data type -> {type(data)}, shape -> {data.shape}\n")

        if self.model is None:
            raise NotImplementedError("\nModel is not trained/loaded.")
        return np.random.randint(0, 10)


class RandomForest(DigitClassificationInterface):
    def __init__(self, model=None):
        self.model = model

    def predict(self, data: np.ndarray) -> int:
        print(f"\nThis is RandomForestModel. Input data type -> {type(data)}, shape -> {data.shape}\n")

        if self.model is None:
            raise NotImplementedError("\nModel is not trained/loaded.")
        return np.random.randint(0, 10)


class RandomModel(DigitClassificationInterface):
    def predict(self, data: np.ndarray) -> int:
        print(f"\nThis is RandomModel. Input data type -> {type(data)}, shape -> {data.shape}")
        return np.random.randint(0, 10)


class DigitClassifier:
    def __init__(self, model_name: str):
        self.model: DigitClassificationInterface
        self.model_name = model_name

        if model_name == 'cnn':
            self.model = CNN()
        elif model_name == 'rf':
            self.model = RandomForest()
        elif model_name == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("\nUnknown model name.")

    def predict(self, image_28x28x1: np.ndarray) -> int:
        transformed_data = self._transform_input(image_28x28x1)
        return self.model.predict(transformed_data)

    def _transform_input(self, image: np.ndarray):
        if self.model_name == 'cnn':
            # Input: tensor 28x28x1
            transform = transforms.ToTensor()
            return transform(image)
        elif self.model_name == 'rf':
            # Input: 1-d numpy array of length 784 (28x28 pixels)
            return image.flatten()
        elif self.model_name == 'rand':
            # Input: 10x10 numpy array, the center crop of the image
            height, width = image.shape[:2]
            start_row = (height - 10) // 2
            start_col = (width - 10) // 2
            end_row = start_row + 10
            end_col = start_col + 10
            return image[start_row:end_row, start_col:end_col]
        else:
            raise ValueError("\nUnknown model name.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nError: Classification algorithm not specified.")
        print("Usage: python task3.py [cnn | rf | rand]")
        sys.exit(1)

    algorithm_name = sys.argv[1].lower()
    try:
        classifier = DigitClassifier(algorithm_name)
        print(f"\nClassifier initialised with algorithm: {algorithm_name.upper()}")

        img = Image.open("4.jpg")
        image = np.array(img)
        result = classifier.predict(image)
        print(f"\nPrediction result: {result}")

    except ValueError as e:
        print(f"Error during initialisation: {e}")
        sys.exit(1)
