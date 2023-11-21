import json
import re
from dataclasses import field
from typing import Union, List, TypedDict, Optional, Dict, Literal
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from llama_index.schema import TextNode
from typing_extensions import Annotated
from pydantic import BaseModel, Field


@dataclass_json
@dataclass
class SlotSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    type: str = field(metadata={"required": False}, default=None)


@dataclass_json
@dataclass
class FrameSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(default_factory=list)


@dataclass_json
@dataclass
class FrameId:
    module: str
    name: str


# This name inside the FrameSchema and SlotSchema is semantic bearing.
# So there should not be overlapping between schema names.
# the key for skills and slots does not need to be.
@dataclass_json
@dataclass
class Schema:
    skills: Dict[str, FrameSchema]
    slots: Dict[str, SlotSchema]


@dataclass_json
@dataclass
class SchemaStore:
    schemas: Dict[str, Schema]

    def __init__(self, schemas: dict[str, Schema]):
        self.schemas = schemas
        self.func_to_module = {skill["name"]: schema for schema in schemas.values() for skill in schema.skills.values()}

    def get_skills(self, frame_id: FrameId):
        return self.schemas[frame_id.module].skills[frame_id.name]

    def get_module(self, func_name):
        return self.func_to_module[func_name]


@dataclass_json
@dataclass
class FrameValue:
    name: str
    arguments: TypedDict


@dataclass_json()
@dataclass
class FrameState:
    module: Schema
    frame: FrameSchema
    activated: list[str]


# How to present context is strictly state tracking implementation dependent.
@dataclass_json
@dataclass
class DialogExpectation(BaseModel):
    context: list[FrameState]


class EntityInstance(BaseModel):
    label: str = Field(description='the canonical form of the instance')
    expressions: List[str] = Field(description="the expressions used to identify this instance.")


class ListRecognizer(BaseModel):
    rec_type: Literal['list']
    name: str = Field(description="language dependent name")
    description: Optional[str] = Field(description="define what is this type for.")
    instances: List[EntityInstance]


class PatternRecognizer(BaseModel):
    rec_type: Literal['pattern']
    name: str = Field(description="language dependent name")
    description: Optional[str] = Field(description="define what is this type for.")
    pattern: str = Field(description="regex pattern to recognize the instance of this entity.")


class SlotRecognizers(BaseModel):
    slots: Dict[str, str] = Field(description="the mapping from slot name to entity name")
    recognizers: Dict[
        str, Annotated[Union[ListRecognizer, PatternRecognizer], Field(discriminator='rec_type')]] = Field(
        description="the name to recognizer")


# owner is not needed if exemplars are listed insider function specs.
class Exemplar(BaseModel):
    owner: str = Field(description="onwer of this exemplar.")
    template: str = Field(description="the example utterance that should trigger the given skill")
    context: Optional[str] = Field(description="the context under which this exemplar works.", default=None)


# There are two different use cases for exemplars:
# During fine-turning, we need both utterance and exemplars.
# During index, we only need exemplars.
class ExemplarStore(TypedDict):
    name: str
    exemplars: List[Exemplar]


# This is need to convert the camel casing to snake casing.
class CamelToSnake:
    pattern = re.compile(r'(?<!^)(?=[A-Z])')

    @classmethod
    def __call__(cls, text):
        return CamelToSnake.pattern.sub('_', text).lower()


def build_nodes_from_exemplar_store(module: str, store: ExemplarStore, nodes: List[TextNode]):
    for label, exemplars in store.items():
        for exemplar in exemplars:
            nodes.append(
                TextNode(
                    text=exemplar.template,
                    id_=str(hash(exemplar.template))[1:13],
                    metadata={"owner": label, "context": exemplar.context, "module": module},
                    excluded_embed_metadata_keys=["owner", "context", "module"]))


if __name__ == "__main__":
    print(json.dumps(ExemplarStore.model_json_schema(), indent=2))
