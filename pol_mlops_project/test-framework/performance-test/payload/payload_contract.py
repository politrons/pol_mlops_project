from abc import ABC, abstractmethod
from typing import Any, Dict

class PayloadContract(ABC):

    @abstractmethod
    def get_payload(self) -> Dict[str, Any]:
       pass

