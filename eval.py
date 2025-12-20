import logging
from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from retriever import configure_retriever
import sys

def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    if k == 0:
        return 0.0
    return sum([1 for doc in retrieved_at_k if doc in relevant]) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum([1 for doc in retrieved_at_k if doc in relevant]) / len(relevant)

def mean_reciprocal_rank(retrieved_list, ground_truth_list):
    mrr_total = 0.0
    for retrieved, relevant in zip(retrieved_list, ground_truth_list):
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                mrr_total += 1.0 / (i + 1)
                break
    return mrr_total / len(retrieved_list)

def evaluate_retrieval(retriever, test_data, k=10):
    precision_scores = []
    recall_scores = []
    all_retrieved = []
    
    for query, ground_truth in test_data.items():
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_ids = [doc.metadata.get("source", "") for doc in retrieved_docs]
        all_retrieved.append(retrieved_ids)
        
        prec = precision_at_k(retrieved_ids, ground_truth, k)
        rec = recall_at_k(retrieved_ids, ground_truth, k)
        precision_scores.append(prec)
        recall_scores.append(rec)
        logging.info(f"Query: '{query}'")
        logging.info(f"Retrieved IDs: {retrieved_ids[:k]}")
        logging.info(f"Ground Truth: {ground_truth}")
        logging.info(f"Precision@{k}: {prec:.2f}, Recall@{k}: {rec:.2f}")
    
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    mrr = mean_reciprocal_rank(all_retrieved, list(test_data.values()))
    return avg_precision, avg_recall, mrr