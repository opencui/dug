import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM, AutoConfig

from opencui import ModelType
from opencui.core.annotation import (CamelToSnake, DialogExpectation, EntityMetas, Exemplar, FrameValue, ListRecognizer,
                                     OwnerMode, ExactMatcher, get_value)
from opencui.core.config import LugConfig
from opencui.core.prompt import (ExtractiveSlotPrompts, YniPrompts, DescriptionPrompts, ExemplarPrompts)
from opencui.core.retriever import (ContextRetriever, load_context_retrievers)
from opencui.inference.schema_parser import load_all_from_directory

# The modes that we will support.
GenerateMode = Enum("GenerateMode", ["desc", "exemplar", "extractive", "nli"])
GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])
YesNoResult = Enum("YesNoResult", ["Affirmative", "Negative", "Indifferent", "Irrelevant"])

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
    generator = None

    @staticmethod
    def build():
        if GeneratorType[LugConfig.get().generator] == GeneratorType.FftGenerator and Generator.generator is None:
            Generator.generator = FftGenerator()
        if GeneratorType[LugConfig.get().generator] == GeneratorType.LoraGenerator and Generator.generator is None:
            Generator.generator = LoraGenerator()
        return Generator.generator

    @staticmethod
    def get_model_type(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        return config.model_type

    @staticmethod
    def from_pretrained(*args, **kwargs):
        config = AutoConfig.from_pretrained(args[0])
        # Check the model type
        print(f"loading model: {args[0]} with type: {config.model_type}")
        if ModelType.normalize(config.model_type) == ModelType.t5:
            return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        if ModelType.normalize(config.model_type) == ModelType.gpt:
            return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @abstractmethod
    def generate(self, input_texts: list[str], mode: GenerateMode = None):
        pass

    def process_return(self, outputs: list[str], input_texts: list[str]):
        if ModelType.normalize(self.model_type) == ModelType.t5:
            return outputs
        if ModelType.normalize(self.model_type) == ModelType.gpt:
            return [output[len(input_texts[index]):] for index, output in enumerate(outputs)]


# This should be desc/exemplar based.
class LoraGenerator(Generator, ABC):
    def __init__(self):
        parts = LugConfig.get().skill_model.split("/")

        desc_model = f"{parts[0]}/desc-{parts[1]}"
        exemplar_model = f"{parts[0]}/exemplar-{parts[1]}"

        skill_config = PeftConfig.from_pretrained(desc_model)

        model_path = skill_config.base_model_name_or_path

        # Is this the right place to clean cache.
        torch.cuda.empty_cache()

        base_model = Generator.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map=LugConfig.get().llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.models = {}

        self.lora_model = PeftModel.from_pretrained(
            base_model, desc_model, adapter_name=GenerateMode.desc.name)

        self.lora_model.load_adapter(
            exemplar_model, adapter_name=GenerateMode.exemplar.name)

        self.lora_model.load_adapter(
            LugConfig.get().extractive_slot_model, adapter_name=GenerateMode.extractive.name)

        if LugConfig.get().nli_model != "":
            self.lora_model.load_adapter(LugConfig.get().nli_model, adapter_name=GenerateMode.nli.name)

        # Move to device
        self.lora_model.to(LugConfig.get().llm_device)
        self.lora_model.eval()

    def generate(self, input_texts: list[str], mode: GenerateMode):
        self.lora_model.set_adapter(mode.name)
        encoding = self.tokenizer(
            input_texts, padding=True, return_tensors="pt"
        ).to(LugConfig.get().llm_device)

        with torch.no_grad():
            peft_outputs = self.lora_model.generate(
                input_ids=encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )

        results = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        return Generator.process_return(results, input_texts)


# Full finetuned generator
class FftGenerator(Generator, ABC):
    def __init__(self):
        # Is this the right place to clean cache.
        torch.cuda.empty_cache()
        self.model = Generator.from_pretrained(
            LugConfig.get().model,
            return_dict=True,
            device_map=LugConfig.get().llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model_type = Generator.get_model_type(LugConfig.get().model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(LugConfig.get().model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Move to device
        self.model.to(LugConfig.get().llm_device)
        self.model.eval()

    def generate(self, input_texts: list[str], mode: GenerateMode):
        print(input_texts)
        encoding = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(LugConfig.get().llm_device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return self.process_return(results, input_texts)


class SkillConverter(ABC):
    @abstractmethod
    def get_skills(self, text) -> list[str]:
        pass

    @abstractmethod
    def grade(self, text, owner, owner_mode, counts_dict):
        pass


def parse_json_from_string(text, default=None):
    try:
        return json.loads(text)
    except ValueError as e:
        return default


# This is used to pick the owner by first accumulate on the exemplars by weight 2
# then accumulate on desc by weight 1.
class SingleOwnerPicker:
    def __init__(self):
        self.counts = defaultdict(int)
        # This we make sure that
        self.modes = [OwnerMode.normal]

    def accumulate(self, flags: list[bool], owners: list[str], weight=2):
        assert len(flags) == len(owners)
        for index, flag in enumerate(flags):
            if flag:
                self.counts[owners[index]] += weight

    def decide(self):
        pairs = list(self.counts.items())
        pairs.sort(key=lambda x: -x[1])
        pairs = list(filter(lambda x: x[1] > 1, pairs))
        return None if len(pairs) == 0 else pairs[0][0]


class ISkillConverter(SkillConverter, ABC):
    def __init__(self, retriever: ContextRetriever, generator):
        self.retrieve = retriever
        self.generator = generator
        self.desc_prompt = DescriptionPrompts[LugConfig.get().skill_prompt]
        self.example_prompt = ExemplarPrompts[LugConfig.get().skill_prompt]
        self.use_exemplar = True
        self.use_desc = True
        assert self.use_desc or self.use_exemplar
        self.matcher = ExactMatcher

    def build_prompts_by_examples(self, text, nodes, to_snake):
        skill_prompts = []
        owners = []
        owner_modes = []

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        exemplars = [
            Exemplar(
                owner=to_snake.encode(node.metadata["owner"]),
                template=node.text,
                owner_mode=node.metadata["owner_mode"]
            )
            for node in nodes
        ]

        for exemplar in exemplars:
            input_dict = {"utterance": text, "template": exemplar.template}
            skill_prompts.append(self.example_prompt(input_dict))
            owners.append(exemplar.owner)
            owner_modes.append(exemplar.owner_mode)

        return skill_prompts, owners, owner_modes

    def build_prompts_by_desc(self, text, skills, to_snake):
        skill_prompts = []
        owners = []

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        # for now, we process it once.
        for skill in skills:
            input_dict = {"utterance": text, "skill": skill}
            skill_prompts.append(self.desc_prompt(input_dict))
            owners.append(skill["name"])
        return skill_prompts, owners

    @staticmethod
    def parse_results(skill_prompts, owners, skill_outputs, owner_modes):
        if LugConfig.get().converter_debug:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(skill_outputs, indent=2))

        flags = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(skill_outputs)
        ]
        return [owners[index] for index, flag in enumerate(flags) if flag]


    @staticmethod
    def accumulate_debug_for_exemplars(preds, nodes, infos):
        assert len(preds) == len(nodes)
        for index in range(len(preds)):
            item = {
                "type": "exemplar",
                "owner": nodes[index].metadata["owner"],
                "text": nodes[index].metadata["template"],
                "result": preds[index]
            }
            infos.append(item)

    @staticmethod
    def accumulate_debug_for_skills(preds, skills, infos):
        assert len(preds) == len(skills)
        for index in range(len(preds)):
            item = {
                "type": "desc",
                "owner": skills[index].name,
                "text": skills[index].description,
                "result": preds[index]
            }
            infos.append(item)



    def get_full_skills(self, text, expectations, debug=False):
        # For now, we only pick one skill
        picker = SingleOwnerPicker()
        skills, nodes = self.retrieve(text, expectations)
        print(f"get_skills for {text} with {len(nodes)} nodes\n")

        debug_infos = []

        # for exemplar
        if self.use_exemplar:
            exemplar_prompts, owners, owner_modes = self.build_prompts_by_examples(text, nodes, CamelToSnake)
            exemplar_outputs = self.generator.generate(exemplar_prompts, GenerateMode.exemplar)

            exemplar_preds = [
                parse_json_from_string(raw_flag, raw_flag)
                for index, raw_flag in enumerate(exemplar_outputs)
            ]
            print(exemplar_preds)
            if debug:
                self.accumulate_debug_for_exemplars(exemplar_preds, nodes, debug_infos)

            picker.accumulate(exemplar_preds, owners, 1)

        # Now we should use the expectation for improve node score, and filtering
        # the contextual template that is not match.

        # for desc
        if self.use_desc:
            desc_prompts, owners = self.build_prompts_by_desc(text, skills, CamelToSnake)

            desc_outputs = self.generator.generate(desc_prompts, GenerateMode.desc)
            desc_preds = [
                parse_json_from_string(raw_flag, None)
                for index, raw_flag in enumerate(desc_outputs)
            ]
            print(desc_preds)
            if debug:
                self.accumulate_debug_for_skills(desc_preds, skills, debug_infos)

            picker.accumulate(desc_preds, owners, 1)

        label = picker.decide()
        return label, list(map(node_to_exemplar, nodes)), debug_infos

    def get_skills(self, text):
        label, node = self.get_full_skills(text)
        return label

    @staticmethod
    def update(preds, truth, counts, skill_prompts, skill_outputs, output=True):
        pairs = list([str(item) for item in zip(preds, truth)])
        if output:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(pairs, indent=2))

        pairs = zip(preds, truth)
        for index, pair in enumerate(pairs):
            if pair[0] != pair[1] and output:
                print(f"At {index}, {skill_prompts[index]} : {skill_outputs[index]}, not correct.")

        pairs = zip(preds, truth)
        for pair in pairs:
            if pair[1] is None:
                continue
            index = 2 if pair[0] else 0
            index += 1 if pair[1] else 0
            counts[index] += 1

    def grade(self, text, owner, owner_mode, count_dict):
        if not self.matcher.is_good_mode(owner_mode):
            return

        picker = SingleOwnerPicker()
        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)

        # for exemplar
        exemplar_prompts, owners, owner_modes = self.build_prompts_by_examples(text, nodes, CamelToSnake)
        exemplar_outputs = self.generator.generate(exemplar_prompts, GenerateMode.exemplar)
        exemplar_preds = [
            parse_json_from_string(raw_flag, raw_flag)
            for index, raw_flag in enumerate(exemplar_outputs)
        ]
        exemplar_truth = [
            self.matcher.agree(owner, owner_mode, lowner, owner_modes[index])
            for index, lowner in enumerate(owners)]

        assert len(exemplar_preds) == len(exemplar_truth)
        picker.accumulate(exemplar_preds, owners, 2)

        # for desc
        desc_prompts, owners = self.build_prompts_by_desc(text, skills, CamelToSnake)
        desc_outputs = self.generator.generate(desc_prompts, GenerateMode.desc)
        desc_preds = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(desc_outputs)
        ]
        desc_truth = [owner == lowner and OwnerMode[owner_mode] == OwnerMode.normal for lowner in owners]
        assert len(desc_preds) == len(desc_truth)

        picker.accumulate(desc_preds, owners, 1)
        counts = count_dict["skill"]
        predicted_owner = picker.decide()
        concrete = count_dict["skills"]
        if predicted_owner == owner and OwnerMode[owner_mode] in picker.modes:
            counts[1] += 1
            debug_output = False
            concrete[owner][1] += 1
        else:
            counts[0] += 1
            debug_output = True
            concrete[owner][0] += 1

        if debug_output:
            print(f"\n\nMade mistakes on: [ {text} ] expecting [{owner}] but get [{predicted_owner}].")
            print(json.dumps(picker.counts))

        # We only output when there is a need for study
        self.update(exemplar_preds, exemplar_truth, count_dict["exemplar"], exemplar_prompts, exemplar_outputs, debug_output)
        self.update(desc_preds, desc_truth, count_dict["desc"], desc_prompts, desc_outputs, debug_output)



def node_to_exemplar(node):
    meta = node.metadata
    result = {
        "type": "exemplar",
        "template": node.text,
        "ownerFrame": CamelToSnake.decode(meta["owner"])
    }

    # The optional information. Kotlin side use label instead of owner_mode.
    if meta["owner_mode"] != "normal":
        result["label"] = meta["owner_mode"]
    if meta["context_frame"] != "":
        result["contextFrame"] = meta["context_frame"]
    if meta["context_slot"] != "":
        result["contextSlot"] = meta["context_slot"]

    return result


class Converter:
    def __init__(
            self,
            retriever: ContextRetriever,
            entity_metas: EntityMetas = None,
            with_arguments=True,
    ):
        self.retrieve = retriever
        self.recognizer = None
        if entity_metas is not None:
            self.recognizer = ListRecognizer(entity_metas)

        self.generator = Generator.build()
        self.slot_prompt = ExtractiveSlotPrompts[LugConfig.get().slot_prompt]
        self.yni_prompt = YniPrompts[LugConfig.get().yni_prompt]
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r"\[([^]]*)\]")
        self.skill_converter = None

        self.skill_converter = ISkillConverter(retriever, self.generator)
        self.yni_results = {"Affirmative", "Negative", "Indifferent", "Irrelevant" }


    # Reference implementation for function calling.
    def understand(self, text: str) -> FrameValue:
        # low level get skill.
        func_name = self.skill_converter.get_skills(text)

        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module
        slot_labels_of_func = module.skills[func_name]["slots"]

        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            values = []
            if self.recognizer is not None:
                values = self.recognizer.extract_values(slot, text)
            slot_input_dict = {"utterance": text, "values": values}
            slot_input_dict.update(module.slots[slot])
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if LugConfig.get().converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.generate(slot_prompts, GenerateMode.extractive)

        if LugConfig.get().converter_debug:
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


    def debug(self, utterance, expectations):
        _, _, debugs = self.skill_converter.get_full_skills(utterance, expectations, True)
        # For now, we assume single intent.

        # TODO: figure out how to handle the multi intention utterance.
        return debugs


    def detect_triggerables(self, utterance, expectations, debug=False):
        func_name, evidence, _ = self.skill_converter.get_full_skills(utterance, expectations, debug)
        # For now, we assume single intent.
        result = {
            "utterance": utterance,
            "evidence": evidence
        }

        if func_name is not None:
            result["owner"] = CamelToSnake.decode(func_name)

        # TODO: figure out how to handle the multi intention utterance.
        return [result]


    def fill_slots(self, text, slots:list[dict[str, str]], candidates:dict[str, list[str]])-> dict[str, str]:
        slot_prompts = []
        for slot in slots:
            name = slot["name"]
            values = get_value(candidates, name, [])
            slot_input_dict = {"utterance": text, "name": name, "candidates": values}
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if LugConfig.get().converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.generate(slot_prompts, GenerateMode.extractive)

        if LugConfig.get().converter_debug:
            print(json.dumps(slot_outputs, indent=2))

        results = {}
        for index, slot in enumerate(slots):
            # TODO(sean): this the source where we know what is the value, while we do not do
            # normalization here, we need to explicitly
            if slot_outputs[index] != "":
                results[slot["name"]] = {"values" : [slot_outputs[index]], "operator": "=="}
        return results

    def inference(self, utterance:str, questions:list[str]) -> list[str]:
        input_prompts = []
        for question in questions:
            # For now, we ignore the language
            input_dict = {"response": utterance, "question": f"{question}."}
            input_prompts.append(self.yni_prompt(input_dict))

        outputs = self.generator.generate(input_prompts, GenerateMode.nli)
        outputs = list(map(lambda x: x if (x in self.yni_results) else "Irrelevant", outputs))

        if LugConfig.get().converter_debug:
            print(f"{input_prompts} {outputs}")
        return outputs

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


def load_converter(module_path, index_path):
    # First load the schema info.
    module_schema, _, _ = load_all_from_directory(module_path)

    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers(module_schema, index_path)

    # Finally build the converter.
    return Converter(context_retriever)
