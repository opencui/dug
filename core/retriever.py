#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import logging
from dataclasses import dataclass

from datasets import Dataset
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode, NodeWithScore
from llama_index import QueryBundle
from converter.lug_config import LugConfig
from core.embedding import EmbeddingStore
from core.annotation import FrameSchema, ModuleSchema

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)


def build_nodes_from_skills(skills: dict[str, FrameSchema]):
    nodes = []
    for label, skill in skills.items():
        desc = skill["description"]
        name = skill["name"]
        nodes.append(
            TextNode(
                text=desc,
                id_=label,
                metadata={"owner": name},
                excluded_embed_metadata_keys=["owner"]))
    return nodes


# This is used to create the retriever so that we can get dynamic exemplars into understanding.
def create_index(base: str, tag: str, nodes: list[TextNode], embedding: BaseEmbedding):
    path = f"{base}/{tag}/"
    # Init download hugging fact model
    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=embedding,
    )

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    try:
        embedding_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            service_context=service_context)

        keyword_index = SimpleKeywordTableIndex(
            nodes,
            storage_context=storage_context,
            service_context=service_context)

        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=path)
        keyword_index.set_index_id("keyword")
        keyword_index.storage_context.persist(persist_dir=path)
    except Exception as e:
        print(str(e))
        shutil.rmtree(path, ignore_errors=True)


def build_desc_index(dsc: ModuleSchema, output: str, embedding: BaseEmbedding):
    desc_nodes = build_nodes_from_skills(dsc.skills)
    create_index(output, "desc", desc_nodes, embedding)


#
# There are four kinds of mode: embedding, keyword, AND and OR.
#
class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(self, path: str, tag: str, topk: int = 8, mode: str = "embedding") -> None:
        """Init params."""
        if mode not in ("embedding", "keyword", "AND", "OR"):
            raise ValueError("Invalid mode.")

        embedding = EmbeddingStore.get_embedding_by_task(tag)
        service_context = ServiceContext.from_defaults(
            llm=None,
            llm_predictor=None,
            embed_model=embedding)
        storage_context = StorageContext.from_defaults(persist_dir=f"{path}/{tag}/")
        embedding_index = load_index_from_storage(
            storage_context,
            index_id="embedding",
            service_context=service_context)
        keyword_index = load_index_from_storage(
            storage_context,
            index_id="keyword",
            service_context=service_context)

        self._vector_retriever = VectorIndexRetriever(
            index=embedding_index,
            similarity_top_k=topk)
        self._keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "embedding":
            retrieve_ids = vector_ids
        elif self._mode == "keyword":
            retrieve_ids = keyword_ids
        elif self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


def dedup_nodes(old_results: list[TextNode]):
    new_results = []
    intents = set()
    for item in old_results:
        intent = item.metadata["owner"]
        if intent not in intents:
            intents.add(intent)
            new_results.append(item)
    return new_results


class CombinedRetriever:
    def __init__(self, module: ModuleSchema, d_retrievers, e_retriever):
        self.module = module
        self.desc_retriever = d_retrievers
        self.exemplar_retriever = e_retriever
        self.nones = ["NONE"]

    def search(self, query):
        # The goal here is to find the combined descriptions and exemplars.
        desc_nodes = [item.node for item in self.desc_retriever.retrieve(query)]
        exemplar_nodes = [item.node for item in self.exemplar_retriever.retrieve(query)]

        #print(desc_nodes)
        #print(exemplar_nodes)

        exemplar_nodes = dedup_nodes(exemplar_nodes)

        all_nodes = exemplar_nodes

        owners = set([item.metadata["owner"] for item in all_nodes])
        for none in self.nones:
            owners.discard(none)

        skills = [self.module.skills[owner] for owner in owners]
        return skills


def load_retrievers(module, path: str):
    return CombinedRetriever(
        module,
        HybridRetriever(path, "desc", LugConfig.desc_retrieve_topk),
        HybridRetriever(path, "exemplar", LugConfig.exemplar_retrieve_topk))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    output = "./index/sgdskill/"
    from finetune.sgd import SGD

    dsc = SGD("/home/sean/src/dstc8-schema-guided-dialogue/")

    LugConfig.embedding_device = "cuda"
    dataset = dsc.build("train")
    # print(compute_hits(dataset, output, 8))
    # print(compute_k(dataset, output, "exemplar"))
