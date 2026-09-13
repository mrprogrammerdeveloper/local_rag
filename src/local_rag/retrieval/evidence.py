from dataclasses import dataclass


@dataclass
class Evidence:
    id: str
    content: str
    source: str
    page_number: int
    reference_numbers: list[int]