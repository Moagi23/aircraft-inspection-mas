class MetaAgent:
    def __init__(self):
        pass

    def get_tasks_and_agents(self):
        return {
            "serial_number": {
                "label": "Serial Number Inspection",
                "agents": ["SerialNumberAgent"]
            },
            "manual_serial": {
                "label": "Manual Serial Entry",
                "agents": ["ManualSerialEntryAgent"]
            },
            "damage_detection": {
                "label": "Damage Detection",
                "agents": ["DamageDetectionAgent"]
            }
        }