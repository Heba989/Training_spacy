# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import os
import datasets
import pandas as pd


logger = datasets.logging.get_logger(__name__)



_DESCRIPTION = """\
The shared task of kayanhr dataset concerns language-independent named entity recognition. We will concentrate on
four types of named entities: persons, locations, organizations and names of miscellaneous entities that do
not belong to the previous three groups.
The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on
a separate line and there is an empty line after each sentence. The first item on each line is a word, the second
a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags
and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only
if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag
B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Note the dataset uses IOB2
tagging scheme, whereas the original dataset uses IOB1.
For more details see https://www.clips.uantwerpen.be/conll2003/ner/ and https://www.aclweb.org/anthology/W03-0419
"""


class KayanConfig(datasets.BuilderConfig):
    """BuilderConfig for KayanHR resume datasets"""

    def __init__(self, **kwargs):
        """BuilderConfig for Kayan.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(KayanConfig, self).__init__(**kwargs)


class Kayan(datasets.GeneratorBasedBuilder):
    """Kayanhr dataset."""

    BUILDER_CONFIGS = [
        KayanConfig(name="KayanResumeData", version=datasets.Version("1.0.0"), description="resume dataset")]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequance(datasets.Value('string')),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-Degree", "I-Degree",
                                "B-Email", "I-Email",
                                "B-GPA", "I-GPA",
                                "B-GPE", "I-GPE",
                                "B-Major", "I-Major",
                                "B-Phone", "I-Phone",
                                "B-Skills","I-Skills",
                                "B-brthdate", "I-brthdate",
                                "B-contratctype", "I-contratctype",
                                "B-courses", "I-courses",
                                "B-gender", "I-gender",
                                "B-hascertificate", "I-hascertificate",
                                "B-languages", "I-languages",
                                "B-location", "I-location",
                                "B-name", "I-name",
                                "B-nationality", "I-nationality",
                                "B-position", "I-position",
                                "B-studiedat", "I-studiedat",
                                "B-summary", "I-summary",
                                "B-workat", "I-workat"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=" ",
            citation="Kayanhr",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = {
            "train": 'train/train_tagged.csv',
            "dev": 'dev/dev_tagged.csv',
            # "test": os.path.join(folder_path, test),
        }
        # downloaded_files = datasets.load_dataset('csv', data_files=data_files)
        downloaded_files = dl_manager.download_and_extract(data_files)
        return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
                # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
            ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        data = pd.read_csv(filepath)
        for i in range(len(data)):
            yield i, {
                "id": str(i),
                "tokens": eval(data["tokens"][i]),
                "ner_tags": eval(data["ner_tags"][i]),
            }

        # with open(filepath, encoding="utf-8") as f:
        #     guid = 0
        #     tokens = []
        #     ner_tags = []
        #     for line in f:
        #         if line.startswith("-DOCSTART-") or line == "" or line == "\n":
        #             if tokens:
        #                 yield guid, {
        #                     "id": str(guid),
        #                     "tokens": tokens,
        #                     "pos_tags": pos_tags,
        #                     "chunk_tags": chunk_tags,
        #                     "ner_tags": ner_tags,
        #                 }
        #                 guid += 1
        #                 tokens = []
        #                 pos_tags = []
        #                 chunk_tags = []
        #                 ner_tags = []
        #         else:
        #             # conll2003 tokens are space separated
        #             splits = line.split(" ")
        #             tokens.append(splits[0])
        #             pos_tags.append(splits[1])
        #             chunk_tags.append(splits[2])
        #             ner_tags.append(splits[3].rstrip())
        #     # last example
        #     if tokens:
        #         yield guid, {
        #             "id": str(guid),
        #             "tokens": tokens,
        #             "pos_tags": pos_tags,
        #             "chunk_tags": chunk_tags,
        #             "ner_tags": ner_tags,
        #         }