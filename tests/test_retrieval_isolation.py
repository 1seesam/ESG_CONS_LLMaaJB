from dataclasses import dataclass

from src.benchmarking.retrieval import RetrievalAdapter


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict


class FakeVectorStore:
    def similarity_search(self, query, **kwargs):
        filt = kwargs.get("filter", {})
        source = filt.get("source")
        if source == "File_rot.txt":
            return [FakeDoc("ROT context", {"source": "File_rot.txt"})]
        return [FakeDoc("BLAU context", {"source": "File_Blau.txt"})]


class FakeRagSystem:
    def __init__(self):
        self.vectorstore = FakeVectorStore()


def test_retrieval_filter_isolation():
    adapter = RetrievalAdapter(FakeRagSystem())
    context, sources = adapter.get_context("frage", filename_filter="File_rot.txt")
    assert "ROT" in context
    assert sources == ["File_rot.txt"]

