from abc import ABC, abstractmethod

analysis_dict = dict()


class Analysis(ABC):
    def __init__(self, key):
        analysis_dict[key] = self

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
