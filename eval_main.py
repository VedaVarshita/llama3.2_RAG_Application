#############
from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from chat_model import create_rag_chain
from logger_config import logger
import sys

def main():
    try:
        logger.info("Loading environment variables...")
        load_dotenv()
        logger.info("Environment variables loaded successfully.")
    except Exception as e:
        logger.error("Failed to load environment variables.", exc_info=True)
        sys.exit(1)

    # Load documents
    try:
        logger.info("Starting to load documents from the directory...")
        docs = load_pdfs_from_directory('rag-data')
        logger.info(f"Successfully loaded {len(docs)} documents.")
    except Exception as e:
        logger.error("Failed to load documents.", exc_info=True)
        sys.exit(1)

    # Split documents into chunks
    try:
        logger.info("Splitting documents into smaller chunks...")
        document_chunks = split_documents(docs)
        logger.info(f"Split documents into {len(document_chunks)} chunks.")
    except Exception as e:
        logger.error("Failed to split documents.", exc_info=True)
        sys.exit(1)

    # Create vector store
    try:
        logger.info("Creating the vector store...")
        vector_store = create_vector_store(document_chunks)
        logger.info("Vector store created successfully.")
    except Exception as e:
        logger.error("Failed to create vector store.", exc_info=True)
        sys.exit(1)
    
    # Evaluate retrieval accuracy
    try:
        from eval import evaluate_retrieval  
        from retriever import configure_retriever

        logger.info("Evaluating retrieval accuracy on test queries...")
        retriever = configure_retriever(vector_store)

        test_data = {
            "What is the main idea behind Multiclass AdaBoost?": ["rag-data/ML/MulticlassAdaBoost.pdf"],
            "Explain the principles of XGBoost.": ["rag-data/ML/XGBoost.pdf"],
            "What optimization techniques are discussed in Efficient Back Propagation?": ["rag-data/ML/EfficientBackProp.pdf"],
            "How does StyleGAN work?": ["rag-data/ML/StyleGAN.pdf"],
            "What role does PCA play in machine learning according to Minka's paper?": ["rag-data/ML/minka-pca.pdf"],
            "How is stochastic gradient boosting applied in machine learning?": ["rag-data/ML/StochasticGradientBoosting.pdf"],
            "What are the challenges discussed in traditional backpropagation methods?": ["rag-data/ML/backprop_old.pdf"],
            "How does LDA perform topic modeling?": ["rag-data/ML/LDA.pdf"],
            "Describe the architecture of BERT.": ["rag-data/ML/BERT.pdf"],
            "What is the significance of Layer Normalization in neural networks?": ["rag-data/ML/LayerNormalization.pdf"],
            "Discuss the applications of Transfer Visual Models.": ["rag-data/ML/TransferVisualModels.pdf"],
            "What is the no free lunch theorem in machine learning?": ["rag-data/ML/Nofreelunch.pdf"],
            "How can word representation be estimated efficiently?": ["rag-data/ML/Efficientestimationofwordrep.pdf"],
            "Why is Naive Bayes considered optimal in some scenarios?": ["rag-data/ML/OptimalityofNaiveBayes.pdf"],
            "What challenges are associated with aligning general language assistants?": ["rag-data/RL/A_General_Language_Assistant_as_a_Laboratory_for_Alignment_v3.pdf"],
            "How does learning to summarize with human feedback work?": ["rag-data/RL/NeurIPS-2020-learning-to-summarize-with-human-feedback-Paper.pdf"],
            "What is the impact of deep reinforcement learning from human preferences?": ["rag-data/RL/Deep_Reinforcement_Learning_from_Human_Preferencesv4.pdf"],
            "How are language models trained to follow instructions with human feedback?": ["rag-data/RL/Training language models to follow instructions with human feedback.pdf"],
            "Explain the main concepts of Proximal Policy Optimization.": ["rag-data/RL/Proximal_Policy_Optimization_Algorithms.pdf"],
            "Why is Layer Normalization essential for modern neural networks?": ["rag-data/ML/LayerNormalization.pdf"]

        }

        avg_precision, avg_recall, mrr = evaluate_retrieval(retriever, test_data, k=5)
        logger.info(f"Average Precision@5: {avg_precision:.2f}")
        logger.info(f"Average Recall@5: {avg_recall:.2f}")
        logger.info(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")
    except Exception as e:
        logger.error("Failed to evaluate retrieval accuracy.", exc_info=True)

    print(f"Average Precision@5: {avg_precision:.2f}")
    print(f"Average Recall@5: {avg_recall:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")

    # Create RAG chain
    try:
        logger.info("Initializing the RAG chain...")
        rag_chain = create_rag_chain(vector_store)
        logger.info("RAG chain initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize the RAG chain.", exc_info=True)
        sys.exit(1)

    print("Hello world!, Type 'exit' to end the conversation.")
    
    while True:
        try:

            user_question = input("You: ")
            
            if user_question.lower() == 'exit':
                logger.info("User exited the chatbot.")
                print("Byee :)")
                break

            response = rag_chain.invoke(user_question)
            logger.info("Processed user question successfully.")
            
            print("Assistant:", response)
        except Exception as e:
            logger.error("An error occurred during chatbot interaction.", exc_info=True)
            print("An error occurred. Please try again.")


if __name__ == "__main__":
    main()
