from abc import ABC, abstractmethod

class Environment(ABC):
    pass

    @abstractmethod
    def get_resources(self):
        pass


class GPUEnvironment(Environment):
    
    def get_resources(self):
        pass

    # def limit_gpu_usage(self):


# class CPUEnvironment(Environment):
