import json
from datasets import Dataset, load_dataset



if __name__ == "__main__":
    path = "./datasets/conllner"
    factory = load_dataset('tner/conll2003')["train"]
    converter = Conll03OneSlotConverter("PER")
    prompted_factory = PromptedFactory(factory, [converter], ["tokens", "tags"])

    tags = ["train", "test", "validation"]
    for tag in tags:
        examples = prompted_factory[tag]
        with open(f"{path}/{tag}.jsonl", "w") as file:
            print(f"there are {len(examples)} examples left for {tag}.")
            for example in examples:
                file.write(f"{json.dumps(example)}\n")
