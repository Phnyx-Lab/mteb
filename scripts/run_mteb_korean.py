"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""

from __future__ import annotations

import logging

import torch
from sentence_transformers import SentenceTransformer

from mteb import MTEB, get_tasks

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

checkpoint_path = "/home/ubuntu/Aman/RAG_Models/embedding/lightning_logs/version_4/checkpoints/epoch=1-step=158.ckpt"
state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
mapped_state_dict = {}
for key, value in state_dict.items():
    mapped_state_dict[key.replace('model.', '0.auto_model.')] = value


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

TASK_LIST_RERANKING = [] #["MIRACLReranking"]

TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "XPQARetrieval",
    # "MIRACLRetrieval",
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

model_list = [
    # "phnyxlab/finetuned-ml-e5-large-nli",
    # "intfloat/multilingual-e5-base",
    # "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large"
    # "upskyy/kf-deberta-multitask",
    # "jhgan/ko-sroberta-nli",
    # "jhgan/ko-sroberta-multitask",
    # "jhgan/ko-sbert-multitask",
    # "jhgan/ko-sbert-nli",
    # "kakaobank/kf-deberta-base",
    # "deliciouscat/kf-deberta-base-nli",
    # "deliciouscat/kf-deberta-base-sts",
    # "BM-K/KoSimCSE-bert-multitask",
    # "BM-K/KoSimCSE-bert",
    # "BM-K/KoSimCSE-roberta",
    # "BM-K/KoSimCSE-roberta-multitask",
    # "paraphrase-multilingual-MiniLM-L12-v2",
    # "distiluse-base-multilingual-cased-v2",
    # "BAAI/bge-m3"
]

for model_name in model_list:
    logger.info(f"Running model: {model_name}")
    model = SentenceTransformer(model_name)
    model.load_state_dict(mapped_state_dict)
    for task in TASK_LIST:
        task = get_tasks(tasks=[task], languages=["kor"])
        logger.info(f"Running task: {task}")
        evaluation = MTEB(tasks=task)
        evaluation.run(model, raise_error=False, verbosity=2, trust_remote_code=True)
