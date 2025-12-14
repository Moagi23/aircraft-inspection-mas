class MetaAgent:
    def __init__(self):
        pass

    def get_tasks_and_agents(self):
        return {
            "serial_number": {
                "label": "Serial Number Inspection",
                "agents": ["SerialNumberAgent"]
            },
            "serial_number_knowledge": {
                "label": "Serial Number Inspection + Knowledge Agents",
                "agents": ["SerialNumberKnowledgeAgent"]
            },
            "scanner": {
                "label": "Scanner (Auto-save)",
                "agents": ["ScannerAgent"]
            },
            "damage_detection": {
                "label": "Damage Detection",
                "agents": ["DamageDetectionAgent"]
            },
            "manual_serial": {
                "label": "Manual Serial Entry",
                "agents": ["ManualSerialEntryAgent"]
            },
        }