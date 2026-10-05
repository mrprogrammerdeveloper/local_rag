from dataclasses import dataclass


@dataclass
class ReferenceAlignmentResult:

    source: str

    reference_number: int

    evidence_text: str

    reference_text: str

    score: float