class KnowledgeAgent:
    def __init__(self):
        # A simulated database of important serial numbers
        self.important_serials = [
            "S04878",
            "11148A",
            "60802657",
            "D00494",
            "227090-01",
            "220110-07",
            "32430",
            "083112030",
            "355862.50",
            "D06366",
            "24809A",
            "60335681",
            "D01436",
            "205968-04",
            "A5CF64090",
            "67150",
            "61169",
            "32891",
            "BEHN-8221",
            "4459A",
            "240589.10",
     ]

    def get_important_serials(self):
        """
        Return a list of important serial numbers for validation.
        """
        return self.important_serials