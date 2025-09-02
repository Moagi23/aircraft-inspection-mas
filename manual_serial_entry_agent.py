# manual_serial_entry_agent.py
import re

class ManualSerialEntryAgent:
    def __init__(self):
        pass

    def validate(self, text: str) -> bool:
        if not text:
            return False
        s = text.strip()
        return bool(re.fullmatch(r"[A-Za-z0-9\-_/\. ]{1,64}", s))