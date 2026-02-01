import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class TraceEvent:
    ts: float
    kind: str
    data: Dict[str, Any]

class Tracer:
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def log(self, kind: str, **data: Any) -> None:
        self.events.append(TraceEvent(ts=time.time(), kind=kind, data=data))

    def dump_json(self) -> str:
        return json.dumps([asdict(e) for e in self.events], ensure_ascii=False, indent=2)

    def print_tail(self, n: int = 10) -> None:
        for e in self.events[-n:]:
            print(f"[{e.kind}] {e.data}")
