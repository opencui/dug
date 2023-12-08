import json
import re
from abc import ABC, abstractmethod

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from opencui_lug.core.annotation import (CamelToSnake, DialogExpectation,
                                         EntityMetas, Exemplar, FrameValue,
                                         ListRecognizer)
from opencui_lug.core.config import LugConfig
from opencui_lug.core.prompt import (BinarySkillPrompts, ExtractiveSlotPrompts,
                                     LayeredPrompts, MulticlassSkillPrompts,
                                     NliPrompts)
from opencui_lug.core.retriever import (ContextRetriever,
                                        load_context_retrievers)
from opencui_lug.inference.schema_parser import load_all_from_directory


# In case you are curious about decoding: https://huggingface.co/blog/how-to-generate
# We are not interested in the variance, so we do not do sampling not beam search.
#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
# Generator is responsible for low level things, we will have two different implementation
# local/s-lora. Converter is built on top of generator.
class Generator(ABC):
    @abstractmethod
    def for_skill(self, input_texts):
        pass

    @abstractmethod
    def for_extractive_slot(self, input_texts):
        pass

    @abstractmethod
    def for_nli(self, input_texts):
        pass


class LocalGenerator(Generator, ABC):
    def __init__(self):
        skill_config = PeftConfig.from_pretrained(LugConfig.skill_model)

        model_path = skill_config.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.lora_model = PeftModel.from_pretrained(
            base_model, LugConfig.skill_model, adapter_name="skill"
        )
        self.lora_model.load_adapter(
            LugConfig.extractive_slot_model, adapter_name="extractive_slot"
        )
        self.lora_model.load_adapter(LugConfig.nli_prompt, adapter_name="nli")

    @classmethod
    def generate(cls, peft_model, peft_tokenizer, input_text):
        peft_encoding = peft_tokenizer(
            input_text, padding=True, return_tensors="pt"
        ).to("cuda:0")
        peft_outputs = peft_model.generate(
            input_ids=peft_encoding.input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=128,
                pad_token_id=peft_tokenizer.eos_token_id,
                eos_token_id=peft_tokenizer.eos_token_id,
                attention_mask=peft_encoding.attention_mask,
                do_sample=False,
                repetition_penalty=1.2,
                num_return_sequences=1,
            ),
        )

        return peft_tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)

    def for_skill(self, input_texts):
        self.lora_model.set_adapter("skill")
        outputs = LocalGenerator.generate(self.lora_model, self.tokenizer, input_texts)
        return [
            output[len(input_texts[index]) :] for index, output in enumerate(outputs)
        ]

    def for_nli(self, input_text):
        self.lora_model.set_adapter("nli")
        output = LocalGenerator.generate(self.lora_model, self.tokenizer, input_text)
        return output[len(input_text) :]

    def for_extractive_slot(self, input_texts):
        self.lora_model.set_adapter("extractive_slot")
        outputs = LocalGenerator.generate(self.lora_model, self.tokenizer, input_texts)
        return [
            output[len(input_texts[index]) :] for index, output in enumerate(outputs)
        ]


class SkillConverter(ABC):
    @abstractmethod
    def get_skill(self, text) -> list[str]:
        pass


def parse_json_from_string(text, default=None):
    try:
        return json.loads(text)
    except ValueError as e:
        return default


class MSkillConverter(SkillConverter):
    def __init__(self, retriever: ContextRetriever, generator=LocalGenerator()):
        self.retrieve = retriever
        self.generator = generator
        self.skill_prompt = MulticlassSkillPrompts[LugConfig.skill_prompt]

    def get_skill(self, text):
        to_snake = CamelToSnake()

        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [
            Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text)
            for node in nodes
        ]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_input_dict = {
            "utterance": text.strip(),
            "examples": exemplars,
            "skills": skills,
        }
        skill_prompt = self.skill_prompt([skill_input_dict])

        skill_outputs = self.generator.for_skill(skill_prompt)

        if LugConfig.converter_debug:
            print(skill_prompt)
            print(skill_outputs)

        func_name = parse_json_from_string(skill_outputs[0])
        if LugConfig.converter_debug:
            print(
                f"{skill_outputs} is converted to {func_name}, valid: {self.retrieve.module.has_module(func_name)}"
            )

        return [func_name]


class BSkillConverter(SkillConverter):
    def __init__(self, retriever: ContextRetriever, generator=LocalGenerator()):
        self.retrieve = retriever
        self.generator = generator
        self.prompt = BinarySkillPrompts[LugConfig.skill_prompt]

    def get_skill(self, text):
        to_snake = CamelToSnake()

        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)
        exemplars = [
            Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text)
            for node in nodes
        ]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_map = {skill["name"]: skill for skill in skills}

        skill_prompts = []
        owners = []
        processed = set()
        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        for o_exemplar in exemplars:
            target = o_exemplar.owner
            # Try not to have more than two examples.
            exemplar_dicts = [
                {
                    "template": exemplar.template,
                    "target": target,
                    "decision": target == exemplar.owner,
                }
                for exemplar in exemplars
            ]

            input_dict = {
                "utterance": text,
                "examples": exemplar_dicts,
                "skill": skill_map[target],
            }
            skill_prompts.append(self.prompt(input_dict))
            owners.append(target)

            processed.add(target)

        for skill in skills:
            if skill["name"] in processed:
                continue
            input_dict = {"utterance": text, "examples": [], "skill": skill}
            skill_prompts.append(self.prompt(input_dict))
            owners.append(skill["name"])

        skill_outputs = self.generator.for_skill(skill_prompts)

        if LugConfig.converter_debug:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(skill_outputs, indent=2))

        flags = [
            parse_json_from_string(raw_flag, False)
            for index, raw_flag in enumerate(skill_outputs)
        ]

        func_names = [owners[index] for index, flag in enumerate(flags) if flag]

        return func_names


class SSkillConverter(SkillConverter):
    def __init__(self, retriever: ContextRetriever, generator=LocalGenerator()):
        self.retrieve = retriever
        self.generator = generator
        self.desc_prompt = LayeredPrompts[LugConfig.skill_prompt][0]
        self.example_prompt = LayeredPrompts[LugConfig.skill_prompt][1]

    def get_skill(self, text):
        to_snake = CamelToSnake()

        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)
        exemplars = [
            Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text)
            for node in nodes
        ]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_prompts = []
        owners = []
        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        for exemplar in exemplars:
            owners.append(exemplar.owner)
            input_dict = {"utterance": text, "template": exemplar.template}
            skill_prompts.append(self.example_prompt(input_dict))

        # for now, we process it once.
        for skill in skills:
            input_dict = {"utterance": text, "skill": skill}
            skill_prompts.append(self.desc_prompt(input_dict))
            owners.append(skill["name"])

        print(skill_prompts)
        skill_outputs = self.generator.for_skill(skill_prompts)

        if LugConfig.converter_debug:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(skill_outputs, indent=2))

        flags = [
            parse_json_from_string(raw_flag, False)
            for index, raw_flag in enumerate(skill_outputs)
        ]

        # We only pick the first function for now, example wins.
        func_names = [owners[index] for index, flag in enumerate(flags) if flag]

        return func_names


class Converter:
    def __init__(
        self,
        retriever: ContextRetriever,
        entity_metas: EntityMetas = None,
        generator=LocalGenerator(),
        with_arguments=True,
    ):
        self.retrieve = retriever
        self.recognizer = None
        if entity_metas is not None:
            self.recognizer = ListRecognizer(entity_metas)

        self.generator = generator
        self.slot_prompt = ExtractiveSlotPrompts[LugConfig.slot_prompt]
        self.nli_prompt = NliPrompts[LugConfig.nli_promt]
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r"\[([^]]*)\]")
        self.skill_converter = None
        if LugConfig.skill_mode == "simple":
            self.skill_converter = SSkillConverter(retriever, generator)
        if LugConfig.skill_mode == "binary":
            self.skill_converter = BSkillConverter(retriever, generator)
        if LugConfig.skill_mode == "multiclass":
            self.skill_converter = MSkillConverter(retriever, generator)
        self.nli_labels = {"entailment": True, "neutral": None, "contradiction": False}

    def understand(
        self, text: str, expectation: DialogExpectation = None
    ) -> FrameValue:
        # low level get skill.
        func_names = self.skill_converter.get_skill(text)

        if len(func_names) == 0:
            return None

        # For now, just return the first one.
        func_name = func_names[0]

        if not self.retrieve.module.has_module(func_name):
            print(f"{func_name} is not recognized.")
            return None
        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module.get_module(func_name)
        slot_labels_of_func = module.skills[func_name]["slots"]
        print(slot_labels_of_func)
        print(module.slots)
        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            values = []
            if self.recognizer is not None:
                values = self.recognizer.extract_values(slot, text)
            slot_input_dict = {"utterance": text, "values": values}
            slot_input_dict.update(module.slots[slot])
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if LugConfig.converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.for_extractive_slot(slot_prompts)

        if LugConfig.converter_debug:
            print(json.dumps(slot_outputs, indent=2))

        slot_values = [parse_json_from_string(seq) for seq in slot_outputs]
        slot_values = dict(zip(slot_labels_of_func, slot_values))
        slot_values = {
            key: value for key, value in slot_values.items() if value is not None
        }

        final_name = func_name
        if module.backward is not None:
            final_name = module.backward[func_name]

        return FrameValue(name=final_name, arguments=slot_values)

    # There are three different
    def decide_boolean(self, utterance, question, lang="en") -> bool:
        # For now, we ignore the language
        input_dict = {"promise": utterance, "hypothesis": f"{question} yes."}
        input_prompt = self.nli_prompt(input_dict)
        output = self.generator.for_nli(input_prompt)
        if LugConfig.converter_debug:
            print(f"{input_prompt} {output}")
        if output not in self.nli_labels:
            return None
        return self.nli_labels[output]

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


def load_converter(module_path, index_path):
    # First load the schema info.
    module_schema, examplers, recognizers = load_all_from_directory(module_path)
    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers(
        {module_path: module_schema}, index_path
    )
    # Finally build the converter.
    return Converter(context_retriever)
