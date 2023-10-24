#!/usr/bin/env python3
import json
from abc import ABC
from datasets import Dataset
from core.commons import SkillInfo, DatasetCreator, SlotInfo, DomainInfo, Exemplar


#
# This is used to create the DatasetCreator from OpenAI function descriptions.
#
# We assume that in each domain, the slot name are unique, and skill name are unique.
#
class OpenAIParser(DatasetCreator, ABC):

    def __init__(self, path) -> None:
        functions = json.load(open(path))
        self.exemplars = []

        skillInfos = {}
        slotInfos = {}
        for func in functions:
            f_name = func["name"]
            f_description = func["description"]
            f_slots = []
            parameters = func["parameters"]
            if parameters["type"] != "object":
                raise RuntimeError("Need to handle this case.")

            for key, slot in parameters["properties"].items():
                f_slots.append(key)
                if key in slotInfos:
                    continue
                else:
                    slot_name = key
                    slot_description = slot["description"]
                    slotInfos[slot_name] = SlotInfo(slot_name, slot_description)
            skillInfos[f_name] = SkillInfo(f_name, f_description, f_slots)
            self.exemplars.extend([ Exemplar(ex, f_name)  for ex in func["exemplars"]])
        self.domain = DomainInfo(skillInfos, slotInfos)

    def build(self, split):
        return Dataset.from_list(self.exemplars)


if __name__ == "__main__":
    openaids = OpenAIParser("./converter/openai_example.json")
    print(openaids.domain)

