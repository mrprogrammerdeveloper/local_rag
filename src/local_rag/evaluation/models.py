from dataclasses import dataclass


@dataclass
class RetrievalTestCase:
    id: str
    question: str
    source: str
    relevant_pages: list[int]


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float