from mteb.abstasks import AbsTaskClassification  # type: ignore
from mteb.abstasks.TaskMetadata import TaskMetadata  # type: ignore


class KorFin(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFin",
        dataset={
            "path": "amphora/korfin-asc-test",
            "revision": "1266fa135fffdf14e7857df3312bf07941f2b17d",
        },
        description="The KorFin-ASC is an extension of KorFin-ABSA, which is a financial sentiment analysis dataset including 8818 samples with (aspect, polarity) pairs annotated. The samples were collected from KLUE-TC and analyst reports from Naver Finance.",
        reference="https://huggingface.co/datasets/amphora/korfin-asc",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=(
            "2022-01-01",
            "2022-12-31",
        ),  # Assumed date based on the citations in the paper
        domains=["News", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY-SA 4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""""
        @article{son2023removing,
        title={Removing Non-Stationary Knowledge From Pre-Trained Language Models for Entity-Level Sentiment Classification in Finance},
        author={Son, Guijin and Lee, Hanwool and Kang, Nahyeon and Hahm, Moonjeong},
        journal={arXiv preprint arXiv:2301.03136},
        year={2023}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 3795},
            "avg_character_length": {"test": 90.99},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"SRC": "text", "SENTIMENT": "label"}
        ).remove_columns(["SID", "TYPE", "ASPECT"])
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
