# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import json
from abc import ABC, abstractmethod
from collections import defaultdict
import random
from pydantic import BaseModel
from torch import Generator

from opendu.core.annotation import (CamelToSnake, Exemplar, FrameSchema)
from opendu.core.config import RauConfig
from opendu.core.matcher import OwnerMode, ExactMatcher
from opendu.core.prompt import (PromptManager, Task)
from opendu.core.retriever import (ContextRetriever)
from opendu.inference.decoder import Decoder, OutputExpectation

from opendu.utils.json_tools import parse_json_from_string
from itertools import islice

class FrameState(BaseModel):
    frame: str
    slot: str
    slotType: str


# How to present context is strictly state tracking implementation dependent.
class DialogExpectation(BaseModel):
    context: list[FrameState]


class BcSkillExample(BaseModel):
    template: str
    label: bool # True for positive, False for negative.


class SkillDemonstration(BaseModel):
    skill: FrameSchema
    exemples: list[BcSkillExample] = []


#
# The intent detector try to detect all triggerable intents from user utterance, with respect to
# existing conversational history, summarized in expectations. However, expectations are only used
# during the retrieval stage, not in the decision stage beyond that.

# Because it is potentially a multi-class problem, we use the binary classification as the
# tradeoff between cost (performance) and flexibility(accuracy).
# Intent dectection cast as single class, binary classification problem.
class BcIntentDetector:
    def __init__(self, retriever: ContextRetriever):
        self.retrieve = retriever
        self.generator = Decoder.get()
        self.debug = RauConfig.get().id_debug
        self.module = retriever.module if retriever else None

    @staticmethod
    def get_closest_template(owner: str, exemplars: list[Exemplar], k: int) -> list[BcSkillExample]:
        positives = list(map(lambda x: BcSkillExample(template=x.template, label=True), islice((x for x in exemplars if x.owner == owner), k)))
        negatives = list(map(lambda x: BcSkillExample(template=x.template, label=False), islice((x for x in exemplars if x.owner != owner), k)))
        if len(positives) == 0 or len(negatives) == 0:
            return []
        else:
            examples = positives + negatives
            random.shuffle(examples)
            return examples

    def build_skills(self, text, skills, exemplars: list[Exemplar]) -> list[SkillDemonstration]:
        skill_metas = []
        for skill in skills:
            print(f"build demo for skill: {skill.name}: {skill}")
            print(skill)
            skill_metas.append(
                SkillDemonstration(
                    skill=skill,
                    exemples= self.get_closest_template(skill.name, exemplars, 1)
                )
            )
        return skill_metas
    
    def detect_intents(self, text:str, candidates:dict[str, list[str]], expectedFrames:list[str]=[]):
        print(f"parse for skill: {text} with {expectedFrames} and {candidates}")
        # For now, we only pick one skill
        # TODO: try to use candidates. 
        skills, exemplar_nodes = self.retrieve(text)
        print(f"get_skills for {text} with {len(exemplar_nodes)} nodes\n")

        debug_infos = []

        exemplars = [self.retrieve.get_exemplar_from_node(node) for node in exemplar_nodes]
        skill_with_exemplars = self.build_skills(text, skills, exemplars)

        # Now we should use the expectation for improve node score, and filtering
        # the contextual template that is not match.
        skill_prompts = []
        build_prompt = PromptManager.get_builder(Task.IdBc, input_mode=True)
        for skill_demonstration in skill_with_exemplars:
            skill_prompts.append(
                build_prompt(
                    {
                        "skill": skill_demonstration.skill,
                        "exemplars": skill_demonstration.exemples,
                        "utterance": text,
                        "arguments": candidates
                    }
                )
            )
        print(f"skill prompt: {skill_prompts}")
        skill_outputs = self.generator.generate(skill_prompts, OutputExpectation(choices=["True", "False"]))
        
        # For now we assume single intent.
        skill_outputs = [output.outputs[0] for output in skill_outputs]
        print(f"skill outputs: {skill_outputs}")
        print(f"skills: {skills}")
        print(f"sizes: {len(skill_outputs)}, {len(skills)}")

        zipped = list(zip(skill_with_exemplars, skill_outputs))
        # Later we can run this twice, first with examples (more trustworthy), then without examples.
        demonstrations = (paired[0] for paired in zipped if paired[1].text == "True")
        print(demonstrations)
        results = []
        for demostration in demonstrations:
            results.append(        # For now, we assume single intent.
                {
                    "owner": demostration.skill.label,
                    "evidence": [],
                    "utterance": text,
                }
            )
        
        if results == []:
            results.append({
                "utterance": text,
                "evidence": [exemplar.model_dump() for exemplar in exemplars]
            })

        # TODO: figure out how to handle the multi intention utterance.
        print(results)
        return results


def node_to_exemplar(node):
    meta = node.metadata
    result = {
        "type": "exemplar",
        "template": meta["template"],
        "ownerFrame": meta["owner"]
    }

    # The optional information. Kotlin side use label instead of owner_mode.
    if meta["owner_mode"] != "normal":
        result["label"] = meta["owner_mode"]
    if meta["context_frame"] != "":
        result["contextFrame"] = meta["context_frame"]
    if meta["context_slot"] != "":
        result["contextSlot"] = meta["context_slot"]

    return result


if __name__ == "__main__":

    skill = FrameSchema(
        name="build_reservation_module",
        description="build a reservation module")

    examples = [
        BcSkillExample(template="can you help me to build a table reservation module", label=True), 
        BcSkillExample(template="I like to reserve a table", label=False)
    ]


    build_prompt = PromptManager.get_builder(Task.IdBc, input_mode=True)
    prompt0 = build_prompt({"utterance": "can you help me to make a table reservation please?", "skill": skill, "examples": examples, "arguments": {}})
    prompt1 = build_prompt({"utterance": "I like to build a table reservation module, please", "skill": skill, "examples": examples, "arguments": {}})
    print(prompt0)

    generator = Decoder.get()

    raw_output = generator.generate([prompt0, prompt1], OutputExpectation(choices=["True", "False"]))
    outputs = [output.outputs for output in raw_output]
    print(outputs)