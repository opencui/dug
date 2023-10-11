from abc import ABC
from datasets import load_dataset, Dataset
from builders.commons import Domain, DatasetCreator, SimplePrompt

# Each raw dataset should be responsible for a couple of things:
# 1. Whether there are new special token need to be added.
# 2. We should be able to access the train, test, and validation, as well as other split for different use case.
# 3. We should be able to modify the way how instance are created, think prompting.

# Each dataset contains one or more domain, and it can create one or more target task/dataset.
# in universal sense so that we can also fine tune them together.
# Each domain has many skills and slots, skills shares these slots. So that at DU level these
# can be defined once and used many times. (Entities are reused at the type level, and slot can be
# reused by implement frames, so it is platform level reuse).


domain = Domain(
    skills=['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation',
            'recommend', 'request_attribute'],
    skill_descriptions=None,
    slots=['name', 'release_year', 'esrb', 'genres', 'platforms', 'available_on_steam', 'has_linux_release',
           'has_mac_release', 'specifier', 'rating', 'player_perspective', 'has_multiplayer', 'developer',
           'exp_release_date'],
    slot_descriptions={
        'name': 'The name of a video game (e.g., Rise of the Tomb Raider).',
        'release_year': 'The year a video game was released in (e.g., 2015).',
        'exp_release_date': 'For a not-yet-released game, the date when it is expected to be released (e.g., February 22, 2019). Note: This slot cannot appear together with release_year in the same dialogue act.',
        'developer': 'The name of the studio/person that created the game (e.g., Crystal Dynamics).',
        'genres': 'A list of one or more genre labels from a set of possible values (e.g., action-adventure, shooter).',
        'player_perspective': 'A list of one or more perspectives from which the game is/can be played (possible values: first person, third person, side view, bird view).',
        'platforms': "A list of one or more gaming platforms the game was officially released for (possible values: PC, PlayStation, Xbox, Nintendo, Nintendo Switch).",
        'esrb': "A game's content rating as determined by the ESRB (possible values: E (for Everyone), E 10+ (for Everyone 10 and Older), T (for Teen), M (for Mature)).",
        'rating': "Depending on the dialogue act this slot is used with, it is a categorical representation of either the game's average rating or the game's liking (possible values: excellent, good, average, poor).",
        'has_multiplayer': "Indicates whether a game supports multiplayer or can only be played in single-player mode (possible values: yes, no).",
        'available_on_steam': "Indicates whether a game can be purchased through the Steam digital distribution service (possible values: yes, no).",
        'has_linux_release': "Indicates whether a game is supported on Linux operating systems (possible values: yes, no).",
        'has_mac_release': "Indicates whether a game is supported on macOS (possible values: yes, no).",
        'specifier': "A game specifier used by the request DA, typically an adjective (e.g., addictive, easiest, overrated, visually impressive)."
    }
)


class Viggo(DatasetCreator):
    def __init__(self, prompt):
        self.prompt = prompt

    def get_meta(self) -> Domain:
        return domain

    def build(self, split: str) -> Dataset:
        datasets = load_dataset("GEM/viggo")
        if split == "eval":
            split = "validation"
        dataset = datasets[split]
        dataset = dataset.rename_column("target", "utterance")
        dataset = dataset.rename_column("meaning_representation", "output")
        dataset = dataset.remove_columns("references")
        dataset = dataset.map(lambda x: {"input": self.prompt(x)})
        return dataset


if __name__ == "__main__":
    prompt = SimplePrompt("Convert the input to structured representation. Input: {{utterance}} Output:")
    viggo = Viggo(prompt)
    print(viggo.build("train"))
