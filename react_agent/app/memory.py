from typing import Any, Dict, List

class Memory:
    """
    最小“短期记忆”：存 items 给 Responses API。
    后面你可以扩展：
    - 长期记忆：向量库 / 文档库
    - 会话摘要：token 变大时压缩
    """
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(item)

    def extend(self, items: List[Dict[str, Any]]) -> None:
        self.items.extend(items)

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.items)
