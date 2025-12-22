"""
Comprehensive evaluation comparing Standard RAG, LangGraph, and DSPy modes
"""

import logging
from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from retriever import configure_retriever
from chat_model import create_rag_chain
import sys
import time
from typing import Dict, List, Tuple

from lg_langGraph import setup_dspy, run_with_langgraph, run_with_dspy, build_rag_workflow
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# RETRIEVAL METRICS (from your original code)
# ============================================

def precision_at_k(retrieved, relevant, k):
    """Calculate precision@k"""
    retrieved_at_k = retrieved[:k]
    if k == 0:
        return 0.0
    return sum([1 for doc in retrieved_at_k if doc in relevant]) / k

def recall_at_k(retrieved, relevant, k):
    """Calculate recall@k"""
    retrieved_at_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum([1 for doc in retrieved_at_k if doc in relevant]) / len(relevant)

def mean_reciprocal_rank(retrieved_list, ground_truth_list):
    """Calculate MRR"""
    mrr_total = 0.0
    for retrieved, relevant in zip(retrieved_list, ground_truth_list):
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                mrr_total += 1.0 / (i + 1)
                break
    return mrr_total / len(retrieved_list)

def evaluate_retrieval(retriever, test_data, k=5):
    """Evaluate retrieval performance"""
    precision_scores = []
    recall_scores = []
    all_retrieved = []
    
    for query, ground_truth in test_data.items():
        retrieved_docs = retriever.invoke(query)  # Updated method
        retrieved_ids = [doc.metadata.get("source", "") for doc in retrieved_docs]
        all_retrieved.append(retrieved_ids)
        
        prec = precision_at_k(retrieved_ids, ground_truth, k)
        rec = recall_at_k(retrieved_ids, ground_truth, k)
        precision_scores.append(prec)
        recall_scores.append(rec)
        
        logger.debug(f"Query: '{query}'")
        logger.debug(f"Retrieved IDs: {retrieved_ids[:k]}")
        logger.debug(f"Ground Truth: {ground_truth}")
        logger.debug(f"Precision@{k}: {prec:.2f}, Recall@{k}: {rec:.2f}")
    
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    mrr = mean_reciprocal_rank(all_retrieved, list(test_data.values()))
    
    return avg_precision, avg_recall, mrr

# ============================================
# GENERATION QUALITY METRICS
# ============================================

def evaluate_answer_quality(answer: str, question: str) -> Dict[str, float]:
    """
    Evaluate answer quality based on various metrics
    """
    metrics = {}
    
    # 1. Length appropriateness (good answers are 50-300 chars)
    answer_length = len(answer)
    if 50 <= answer_length <= 300:
        metrics['length_score'] = 1.0
    elif answer_length < 50:
        metrics['length_score'] = answer_length / 50.0
    else:
        metrics['length_score'] = max(0.5, 300 / answer_length)
    
    # 2. Contains keywords from question
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    keyword_overlap = len(question_words & answer_words) / len(question_words)
    metrics['keyword_relevance'] = keyword_overlap
    
    # 3. Not a refusal/error message
    refusal_phrases = [
        "i don't know",
        "i don't have",
        "cannot answer",
        "not enough information",
        "error"
    ]
    is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
    metrics['confidence_score'] = 0.0 if is_refusal else 1.0
    
    # 4. Overall score (weighted average)
    metrics['overall_score'] = (
        0.3 * metrics['length_score'] +
        0.3 * metrics['keyword_relevance'] +
        0.4 * metrics['confidence_score']
    )
    
    return metrics

def evaluate_generation(rag_function, test_data: Dict[str, List[str]], mode_name: str):
    """
    Evaluate generation quality for a given RAG mode
    """
    results = {
        'mode': mode_name,
        'answers': [],
        'metrics': [],
        'latencies': []
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {mode_name}")
    logger.info(f"{'='*60}")
    
    for question, expected_sources in test_data.items():
        try:
            start_time = time.time()
            answer = rag_function(question)
            latency = time.time() - start_time
            
            metrics = evaluate_answer_quality(answer, question)
            
            results['answers'].append({
                'question': question,
                'answer': answer,
                'expected_sources': expected_sources,
                'latency': latency
            })
            results['metrics'].append(metrics)
            results['latencies'].append(latency)
            
            logger.info(f"\nQ: {question[:60]}...")
            logger.info(f"A: {answer[:100]}...")
            logger.info(f"Quality: {metrics['overall_score']:.2f} | Latency: {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            results['answers'].append({
                'question': question,
                'answer': f"ERROR: {str(e)}",
                'expected_sources': expected_sources,
                'latency': 0
            })
            results['metrics'].append({
                'overall_score': 0.0,
                'length_score': 0.0,
                'keyword_relevance': 0.0,
                'confidence_score': 0.0
            })
            results['latencies'].append(0)
    
    # Calculate aggregate metrics
    results['avg_quality'] = sum(m['overall_score'] for m in results['metrics']) / len(results['metrics'])
    results['avg_latency'] = sum(results['latencies']) / len(results['latencies'])
    results['success_rate'] = sum(m['confidence_score'] for m in results['metrics']) / len(results['metrics'])
    
    return results

# ============================================
# MAIN EVALUATION
# ============================================

def main():
    # Load environment and setup
    load_dotenv()
    
    # Load and process documents
    logger.info("Loading documents...")
    docs = load_pdfs_from_directory('rag-data')
    document_chunks = split_documents(docs)
    vector_store = create_vector_store(document_chunks)
    
    # Setup components
    retriever = configure_retriever(vector_store)
    rag_chain = create_rag_chain(vector_store)
    dspy_enabled = setup_dspy()
    
    # Test data
    # test_data = {
    #     "What is the main idea behind Multiclass AdaBoost?": ["rag-data/ML/MulticlassAdaBoost.pdf"],
    #     "Explain the principles of XGBoost.": ["rag-data/ML/XGBoost.pdf"],
    #     "How does StyleGAN work?": ["rag-data/ML/StyleGAN.pdf"],
    #     "Describe the architecture of BERT.": ["rag-data/ML/BERT.pdf"],
    #     "What is the significance of Layer Normalization?": ["rag-data/ML/LayerNormalization.pdf"],
    #     "Explain the main concepts of Proximal Policy Optimization.": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
    #     "What is PPO?": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
    #     "How does LDA perform topic modeling?": ["rag-data/ML/LDA.pdf"],
    # }

    test_data = {
        # Machine Learning - Boosting Algorithms
        "What is the main idea behind Multiclass AdaBoost?": ["rag-data/ML/MulticlassAdaBoost.pdf"],
        "Explain the principles of XGBoost.": ["rag-data/ML/XGBoost.pdf"],
        "How is stochastic gradient boosting applied in machine learning?": ["rag-data/ML/StochasticGradientBoosting.pdf"],
        
        # Machine Learning - Neural Networks
        "What optimization techniques are discussed in Efficient Back Propagation?": ["rag-data/ML/EfficientBackProp.pdf"],
        "What are the challenges discussed in traditional backpropagation methods?": ["rag-data/ML/backprop_old.pdf"],
        "What is the significance of Layer Normalization in neural networks?": ["rag-data/ML/LayerNormalization.pdf"],
        "Why is Layer Normalization essential for modern neural networks?": ["rag-data/ML/LayerNormalization.pdf"],
        
        # Machine Learning - Generative Models
        "How does StyleGAN work?": ["rag-data/ML/StyleGAN.pdf"],
        "What are the key innovations in StyleGAN architecture?": ["rag-data/ML/StyleGAN.pdf"],
        
        # Machine Learning - NLP & Transformers
        "Describe the architecture of BERT.": ["rag-data/ML/BERT.pdf"],
        "How can word representation be estimated efficiently?": ["rag-data/ML/Efficientestimationofwordrep.pdf"],
        "What makes BERT different from previous language models?": ["rag-data/ML/BERT.pdf"],
        
        # Machine Learning - Dimensionality Reduction
        "What role does PCA play in machine learning according to Minka's paper?": ["rag-data/ML/minka-pca.pdf"],
        "Explain the mathematical foundations of PCA.": ["rag-data/ML/minka-pca.pdf"],
        
        # Machine Learning - Topic Modeling
        "How does LDA perform topic modeling?": ["rag-data/ML/LDA.pdf"],
        "What are the applications of Latent Dirichlet Allocation?": ["rag-data/ML/LDA.pdf"],
        
        # Machine Learning - Classical Methods
        "Why is Naive Bayes considered optimal in some scenarios?": ["rag-data/ML/OptimalityofNaiveBayes.pdf"],
        "What is the no free lunch theorem in machine learning?": ["rag-data/ML/Nofreelunch.pdf"],
        
        # Machine Learning - Transfer Learning
        "Discuss the applications of Transfer Visual Models.": ["rag-data/ML/TransferVisualModels.pdf"],
        "How does transfer learning improve model performance?": ["rag-data/ML/TransferVisualModels.pdf"],
        
        # Reinforcement Learning - Policy Optimization
        "Explain the main concepts of Proximal Policy Optimization.": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
        "What is PPO?": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
        "What are the advantages of PPO over other policy gradient methods?": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
        
        # Reinforcement Learning - Human Feedback
        "How does learning to summarize with human feedback work?": ["rag-data/RL/NeurIPS-2020-learning-to-summarize-with-human-feedback-Paper.pdf"],
        "What is the impact of deep reinforcement learning from human preferences?": ["rag-data/RL/Deep_Reinforcement_Learning_from_Human_Preferencesv4.pdf"],
        "How are language models trained to follow instructions with human feedback?": ["rag-data/RL/Training language models to follow instructions with human feedback.pdf"],
        "What challenges are associated with aligning general language assistants?": ["rag-data/RL/A_General_Language_Assistant_as_a_Laboratory_for_Alignment_v3.pdf"],
        
        # Short/Ambiguous Questions (testing robustness)
        "What is AdaBoost?": ["rag-data/ML/MulticlassAdaBoost.pdf"],
        "Explain gradient boosting": ["rag-data/ML/StochasticGradientBoosting.pdf", "rag-data/ML/XGBoost.pdf"],
        "What is backpropagation?": ["rag-data/ML/EfficientBackProp.pdf", "rag-data/ML/backprop_old.pdf"],
        "How does PCA work?": ["rag-data/ML/minka-pca.pdf"],
        "What is LDA?": ["rag-data/ML/LDA.pdf"],
        
        # Complex/Multi-hop Questions (testing advanced reasoning)
        "Compare XGBoost and traditional gradient boosting methods.": ["rag-data/ML/XGBoost.pdf", "rag-data/ML/StochasticGradientBoosting.pdf"],
        "How do modern optimization techniques improve upon traditional backpropagation?": ["rag-data/ML/EfficientBackProp.pdf", "rag-data/ML/backprop_old.pdf"],
        "What are the connections between PPO and human feedback in RL?": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf", "rag-data/RL/Deep_Reinforcement_Learning_from_Human_Preferencesv4.pdf"],
    }
    
    # ============================================
    # PART 1: RETRIEVAL EVALUATION
    # ============================================
    logger.info("\n" + "="*60)
    logger.info("PART 1: RETRIEVAL QUALITY EVALUATION")
    logger.info("="*60)
    
    retrieval_results = evaluate_retrieval(retriever, test_data, k=5)
    
    print("\n RETRIEVAL METRICS:")
    print(f"  Precision@5: {retrieval_results[0]:.2%}")
    print(f"  Recall@5:    {retrieval_results[1]:.2%}")
    print(f"  MRR:         {retrieval_results[2]:.2%}")
    
    # ============================================
    # PART 2: GENERATION EVALUATION
    # ============================================
    logger.info("\n" + "="*60)
    logger.info("PART 2: GENERATION QUALITY EVALUATION")
    logger.info("="*60)
    
    all_results = []
    
    # Mode 1: Standard RAG
    mode1_results = evaluate_generation(
        lambda q: rag_chain.invoke(q),
        test_data,
        "Standard RAG"
    )
    all_results.append(mode1_results)
    
    # Mode 2: LangGraph
    mode2_results = evaluate_generation(
        lambda q: run_with_langgraph(q, retriever, rag_chain),
        test_data,
        "LangGraph"
    )
    all_results.append(mode2_results)
    
    # Mode 3: DSPy (if available)
    if dspy_enabled:
        mode3_results = evaluate_generation(
            lambda q: run_with_dspy(q, retriever),
            test_data,
            "DSPy"
        )
        all_results.append(mode3_results)
    
    # ============================================
    # FINAL COMPARISON
    # ============================================
    print("\n" + "="*60)
    print(" FINAL COMPARISON")
    print("="*60)
    
    print("\n Retrieval Performance:")
    print(f"  Precision@5: {retrieval_results[0]:.2%}")
    print(f"  Recall@5:    {retrieval_results[1]:.2%}")
    print(f"  MRR:         {retrieval_results[2]:.2%}")
    
    print("\n Generation Performance:")
    print(f"\n{'Mode':<15} {'Quality':<10} {'Success Rate':<15} {'Avg Latency':<12}")
    print("-" * 60)
    
    for result in all_results:
        print(f"{result['mode']:<15} "
              f"{result['avg_quality']:.2%}"
              f"{result['success_rate']:.2%}"
              f"{result['avg_latency']:.2f}s")
    
    # Determine winner
    print("\n WINNER:")
    best_mode = max(all_results, key=lambda x: x['avg_quality'])
    print(f"  Best Overall: {best_mode['mode']} (Quality: {best_mode['avg_quality']:.2%})")
    
    fastest_mode = min(all_results, key=lambda x: x['avg_latency'])
    print(f"  Fastest:      {fastest_mode['mode']} (Latency: {fastest_mode['avg_latency']:.2f}s)")
    
    most_reliable = max(all_results, key=lambda x: x['success_rate'])
    print(f"  Most Reliable: {most_reliable['mode']} (Success: {most_reliable['success_rate']:.2%})")
    
    # Save detailed results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'retrieval': {
                'precision_at_5': retrieval_results[0],
                'recall_at_5': retrieval_results[1],
                'mrr': retrieval_results[2]
            },
            'generation': [
                {
                    'mode': r['mode'],
                    'avg_quality': r['avg_quality'],
                    'avg_latency': r['avg_latency'],
                    'success_rate': r['success_rate'],
                    'answers': r['answers']
                }
                for r in all_results
            ]
        }, f, indent=2)
    
    print("\n Detailed results saved to 'evaluation_results.json'")

if __name__ == "__main__":
    main()