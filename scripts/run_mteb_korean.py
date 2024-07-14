"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB, get_tasks

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
    "KLUE-TC",
    "KorFin",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultilingualSentimentClassification",
    "SIB200Classification",
]

TASK_LIST_CLUSTERING = ["SIB200ClusteringS2S"]

TASK_LIST_PAIR_CLASSIFICATION = ["KLUE-NLI", "PawsXPairClassification"]

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "XPQARetrieval",
]

TASK_LIST_STS = ["KLUE-STS", "KorSTS", "STS17"]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

for task in TASK_LIST:
    task = get_tasks(tasks=[task], languages=["kor"])
    logger.info(f"Running task: {task}")
    evaluation = MTEB(tasks=task)
    evaluation.run(model)
