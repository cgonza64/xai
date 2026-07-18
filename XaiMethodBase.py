class XaiMethodBase:
    """
    Base class for all XAI methods.
    """

    def __init__(self, model):
        self.name = self.__class__.__name__
        self.model = model

    def explain(self, input_data):
        """
        Generate explanations for the given input data.

        Args:
            input_data: The input data for which explanations are to be generated.

        Returns:
            Explanations for the input data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def visualize(self, input_tensor, target_class, save_path=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def importance_maps(self, explanations):
        """
        Convert explanations to pixel importance maps.
        """
        raise NotImplementedError("Subclasses should implement this method.")