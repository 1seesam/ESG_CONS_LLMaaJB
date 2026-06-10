from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import logging as transformers_logging

from src.chroma_client import create_persistent_client

DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class RAGSystem:
    def __init__(self, db_folder: str = DB_FOLDER, k: int = 5):
        self.db_folder = db_folder
        self.k = max(1, int(k))
        transformers_logging.set_verbosity_error()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        client = create_persistent_client(self.db_folder)
        self.vectorstore = Chroma(
            collection_name="langchain",
            persist_directory=self.db_folder,
            embedding_function=self.embedding_model,
            client=client,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

    def get_context(self, query, filename_filter=None):
        search_kwargs = {"k": self.k}
        if filename_filter:
            search_kwargs["filter"] = {"source": filename_filter}

        docs = self.vectorstore.similarity_search(query, **search_kwargs)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        return context_text

