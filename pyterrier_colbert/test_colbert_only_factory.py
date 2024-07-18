import os
import torch
import pandas as pd
import pyterrier as pt

# 先初始化 PyTerrier
if not pt.started():
    pt.init()

assert pt.started(), "please run pt.init() before importing pyt_colbert"

from pyterrier import tqdm
from pyterrier.datasets import Dataset
from typing import Union, Tuple
from colbert.evaluation.load_model import load_model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.checkpoint import Checkpoint
from warnings import warn

# 加载数据集
dataset = pt.get_dataset("vaswani")
topics = dataset.get_topics().head(50)
qrels = dataset.get_qrels()

# 定义模型路径
colbert_model_path = '/mnt/c/Users/DJH/Desktop/code/colbertv2.0'

class Object:
    pass

class ColBERTModelOnlyFactory:

    def __init__(self, colbert_model: Union[str, Tuple[ColBERT, dict]], gpu=True, mask_punctuation=False, dim=128):
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = dim  
        args.bsize = 128
        args.similarity = 'cosine'        
        args.amp = True
        args.nprobe = 10
        args.part_range = None
        args.mask_punctuation = mask_punctuation
        args.gpu = gpu  # 添加 gpu 属性

        self.gpu = gpu
        if not gpu:
            warn("Gpu disabled, YMMV")
            import colbert.parameters
            colbert.parameters.DEVICE = torch.device("cpu")

        if isinstance(colbert_model, str):
            args.checkpoint = colbert_model
            colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(colbert_model))
            args.colbert = Checkpoint(name=args.checkpoint, colbert_config=colbert_config)
        else:
            assert isinstance(colbert_model, tuple)
            args.colbert, args.checkpoint = colbert_model
            assert isinstance(args.colbert, ColBERT)
            assert isinstance(args.checkpoint, dict)
        
        args.inference = args.colbert
        self.args = args

    def query_encoder(self, detach=True) -> pt.Transformer:
        def _encode_query(row):
            with torch.no_grad():
                Q = self.args.inference.queryFromText([row.query], bsize=512)
                if detach:
                    Q = Q.cpu()
                print(f"Encoded query: {Q[0]}")  # 打印调试信息
                return pd.Series([Q[0]])
            
        def row_apply(df):
            if "docno" in df.columns or "docid" in df.columns:
                warn("You are query encoding an R dataframe, the query will be encoded for each row")
            df["query_embs"] = df.apply(_encode_query, axis=1)
            return df
        
        return pt.apply.generic(row_apply)

    def text_scorer(self, query_encoded=False, doc_attr="text", verbose=False) -> pt.Transformer:
        # def slow_rerank_with_qembs(args, qembs, pids, passages, gpu=True):
        #     inference = args.inference
        #     Q = torch.unsqueeze(qembs, 0)
        #     if gpu:
        #         Q = Q.cuda()
        #     print(f"Query embeddings: {Q}")  # 打印调试信息

        #     D = inference.docFromText(passages, bsize=args.bsize, keep_dims=True, to_cpu=not gpu)
        #     print(f"Document embeddings result: {D}")  # 打印调试信息

        #     if gpu:
        #         D = D.cuda()
        #     scores = (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        #     print(f"Scores: {scores}")  # 打印调试信息

        #     scores = scores.sort(descending=True)
        #     ranked = scores.indices.tolist()
        #     ranked_scores = scores.values.tolist()
        #     ranked_pids = [pids[position] for position in ranked]
        #     ranked_passages = [passages[position] for position in ranked]
        #     return list(zip(ranked_scores, ranked_pids, ranked_passages))
        def slow_rerank_with_qembs(args, qembs, pids, passages, gpu=True):
            inference = args.inference  # 使用已经初始化的推理对象
            print(f"Using inference object: {inference}")

            Q = torch.unsqueeze(qembs, 0)
            if gpu:
                Q = Q.cuda()
            Q = Q.half()  # 将查询嵌入转换为 float16 类型
            print(f"Query embeddings: {Q}")  # 打印调试信息

            D_tuple = inference.docFromText(passages, bsize=args.bsize, keep_dims=True, to_cpu=not gpu)
            print(f"Document embeddings result: {D_tuple}")  # 打印调试信息

            # 解开返回的元组，获取实际的文档嵌入
            D = D_tuple[0]  # 假设返回的元组第一个元素是我们需要的文档嵌入
            if gpu:
                D = D.cuda()
            print(f"Document embeddings after GPU transfer: {D}")  # 打印调试信息

            try:
                scores = (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
                print(f"Scores: {scores}")  # 打印调试信息
            except Exception as e:
                print(f"Error calculating scores: {e}")
                return []

            scores = scores.sort(descending=True)
            ranked = scores.indices.tolist()
            ranked_scores = scores.values.tolist()
            ranked_pids = [pids[position] for position in ranked]
            ranked_passages = [passages[position] for position in ranked]
            return list(zip(ranked_scores, ranked_pids, ranked_passages))
        def _text_scorer(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr = []
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    query = group["query"].values[0]
                    ranking = slow_rerank(self.args, query, group["docno"].values, group[doc_attr].values.tolist())
                    for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        def _text_scorer_qembs(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr = []
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    qembs = group["query_embs"].values[0]
                    query = group["query"].values[0]
                    ranking = slow_rerank_with_qembs(self.args, qembs, group["docno"].values, group[doc_attr].values.tolist(), gpu=self.gpu)
                    for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        return pt.apply.generic(_text_scorer_qembs if query_encoded else _text_scorer)

# 实例化 ColBERTModelOnlyFactory 并测试 rerank
factory = ColBERTModelOnlyFactory(colbert_model_path, gpu=False)  # 设为 False 以在 CPU 上运行

# 编码查询
encoded_queries = factory.query_encoder()(topics)

# 示例文档集
documents = [
    {"docno": "1", "text": "COVID-19 origin traced back to bats in China."},
    {"docno": "2", "text": "Origins of COVID-19 have been intensely debated."},
    {"docno": "3", "text": "The origin of the COVID-19 virus is still unknown."}
]
documents_df = pd.DataFrame(documents)

# 合并查询和文档数据
queries_and_docs = encoded_queries.merge(documents_df, how="cross")

# 进行 rerank
reranked_results = factory.text_scorer(query_encoded=True)(queries_and_docs)

print(reranked_results)
