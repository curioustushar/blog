"""Document ingestion and corpus creation."""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import json
import uuid
import logging

log = logging.getLogger(__name__)


@dataclass
class Document:
    """A single document in the corpus."""
    doc_id: str
    source: str  # wikipedia, books, code, math
    split: str   # train, eval, validation
    text: str
    metadata: Dict


def create_sample_corpus(output_dir: Path, corpus_sizes: Dict[str, int]) -> List[Document]:
    """Create a small sample corpus for demonstration.
    
    Args:
        output_dir: Directory to save corpus
        corpus_sizes: Dict mapping source_split to count
        
    Returns:
        List of created documents
    """
    documents = []
    
    # Wikipedia train samples
    for i in range(corpus_sizes.get("wikipedia_train", 0)):
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source="wikipedia",
            split="train",
            text=f"Wikipedia article {i}. The capital of France is Paris. It is known for the Eiffel Tower. " * 5,
            metadata={"language": "en", "topic": "geography"}
        )
        documents.append(doc)
    
    # Wikipedia eval samples
    for i in range(corpus_sizes.get("wikipedia_eval", 0)):
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source="wikipedia",
            split="eval",
            text=f"Wikipedia eval article {i}. Photosynthesis is the process by which plants convert light into energy. " * 5,
            metadata={"language": "en", "topic": "science"}
        )
        documents.append(doc)
    
    # Books train samples
    for i in range(corpus_sizes.get("books_train", 0)):
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source="books",
            split="train",
            text=f"Book excerpt {i}. Once upon a time, there was a kingdom by the sea. The waves crashed against the shore. " * 5,
            metadata={"genre": "fiction", "author": f"author_{i % 10}"}
        )
        documents.append(doc)
    
    # Code train samples
    for i in range(corpus_sizes.get("code_train", 0)):
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source="code",
            split="train",
            text=f"# Python code example {i}\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n" * 3,
            metadata={"language": "python", "repo": f"repo_{i % 20}"}
        )
        documents.append(doc)
    
    # Math train samples
    for i in range(corpus_sizes.get("math_train", 0)):
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source="math",
            split="train",
            text=f"Math problem {i}. If 3 apples cost 12 dollars, what is the cost of 5 apples? Solution: 12/3 = 4 per apple. 5*4 = 20 dollars. " * 3,
            metadata={"difficulty": i % 3, "topic": "arithmetic"}
        )
        documents.append(doc)
    
    # Save corpus
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "corpus.jsonl"
    
    with open(corpus_file, "w") as f:
        for doc in documents:
            f.write(json.dumps({
                "doc_id": doc.doc_id,
                "source": doc.source,
                "split": doc.split,
                "text": doc.text,
                "metadata": doc.metadata
            }) + "\n")
    
    log.info(f"Created corpus: {len(documents)} documents")
    log.info(f"  Wikipedia train: {corpus_sizes.get('wikipedia_train', 0)}")
    log.info(f"  Wikipedia eval: {corpus_sizes.get('wikipedia_eval', 0)}")
    log.info(f"  Books train: {corpus_sizes.get('books_train', 0)}")
    log.info(f"  Code train: {corpus_sizes.get('code_train', 0)}")
    log.info(f"  Math train: {corpus_sizes.get('math_train', 0)}")
    
    return documents


def load_corpus(corpus_file: Path) -> List[Document]:
    """Load corpus from file."""
    documents = []
    with open(corpus_file) as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(**data))
    return documents
