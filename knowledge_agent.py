class KnowledgeAgent:
    def __init__(self):
        # A simulated database of important serial numbers
        self.important_serials = [
            "A5CF64090",
            "DEF456",
            "XYZ789",
            "SN202501",
            "CRIT998"
        ]

    def get_important_serials(self):
        """
        Return a list of important serial numbers for validation.
        """
        return self.important_serials