from abc import ABC
from typing import Any, List

from llama_index.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings import HuggingFaceEmbedding

from converter.lug_config import LugConfig
from transformers import AutoTokenizer, AutoModel


# We reuse the underlying embedding when we can.
class EmbeddingStore:
    _models: dict[str, SentenceTransformer] = {}
    # We need different instruction pairs for different use cases.
    INSTRUCTIONS = {
        "qa": {

            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },

        "desc": {
            "query": "Convert this example into vector to look for useful function: ",
            "key": "Convert this function description into vector for retrieval: ",
        },

        "exemplar": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },

        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
        "baai_desc": {
            "query": "",
            "key": "Represent this sentence for searching relevant passages:"
        },
        "baai_exemplars": {
            "query": "",
            "key": ""
        }
    }

    @classmethod
    def get_model(cls, model_name):
        if model_name in EmbeddingStore._models:
            return EmbeddingStore._models[model_name]
        else:
            model = SentenceTransformer(model_name, device=LugConfig.embedding_device)
            EmbeddingStore._models[model_name] = model
            return model

    @classmethod
    def get_embedding_by_task(cls, kind):
        model = EmbeddingStore.get_model(LugConfig.embedding_model)
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])

    @classmethod
    def for_description(cls) -> BaseEmbedding:
        model = EmbeddingStore.get_model(LugConfig.embedding_model)
        kind = LugConfig.embedding_desc_prompt
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])

    @classmethod
    def for_exemplar(cls) -> BaseEmbedding:
        model = EmbeddingStore.get_model(LugConfig.embedding_model)
        kind = LugConfig.embedding_desc_prompt
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])


class InstructedEmbeddings(BaseEmbedding):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()

    def __init__(
            self,
            model: SentenceTransformer,
            instruction: dict[str, str],
            **kwargs: Any,
    ) -> None:
        self._model = model
        self._instructions = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    # embedding might have two different modes: one for query, and one for key/text.

    def expand_for_content(self, query):
        return f"{self._instructions['key']} {query}"

    def expand_for_query(self, query):
        return f"{self._instructions['query']} {query}"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(self.expand_for_query(query), normalize_embeddings=True)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(self.expand_for_content(text), normalize_embeddings=True)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [self._instructions["key"] + key for key in texts]
        embeddings = self._model.encode(texts)
        return embeddings.tolist()


class HFEmbeddingStore:
    query_encoder = HuggingFaceEmbedding(model_name="facebook/dragon-plus-query-encoder", device=LugConfig.embedding_device)
    context_encoder = HuggingFaceEmbedding(model_name="facebook/dragon-plus-context-encoder", device=LugConfig.embedding_device)

    @classmethod
    def get_embedding_by_task(cls, kind):
        return EmbeddingStore.for_exemplar() if kind != "desc" else EmbeddingStore.for_description()

    @classmethod
    def for_description(cls) -> BaseEmbedding:
        return DualHFEmbedding(EmbeddingStore.query_encoder, EmbeddingStore.context_encoder)

    @classmethod
    def for_exemplar(cls) -> BaseEmbedding:
        return DualHFEmbedding(EmbeddingStore.query_encoder, EmbeddingStore.query_encoder)


class DualHFEmbedding(BaseEmbedding):
    _query_encoder: HuggingFaceEmbedding = PrivateAttr()
    _context_encoder: HuggingFaceEmbedding = PrivateAttr()

    def __init__(self, query, context, **kwargs: Any):
        self._query_encoder = query
        self._context_encoder = context
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "DualHFEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._query_encoder.get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._context_encoder.get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._context_encoder.get_text_embedding_batch(texts)
