from abc import ABC, abstractmethod

class ModelContract(ABC):

    @abstractmethod
    def get_model_algorithm(self):
       pass

    @abstractmethod
    def log_model(self,model, model_name, signature, input_example):
        pass