from dataclasses import dataclass


@dataclass
class DocumentPage:
    content: str
    page_number: int
    source: str
    
@dataclass
class DocumentChunk:
    content: str
    page_number: int
    source: str
    chunk_index: int

@dataclass
class DocumentReference:
    number: int
    content: str
    source: str