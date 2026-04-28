# AutoResolve: NLP Triage & RAG Support Agent 🤖

AutoResolve is a sophisticated AI Agent designed to automate enterprise IT support ticket triage. It combines custom Intent Classification, Retrieval-Augmented Generation (RAG), and a Generative LLM to provide instant, policy-accurate solutions to user queries.

### 🔗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Shauryaaa05/AutoResolve-Agent)

##  Key Features
* **Intent Classification:** Uses a fine-tuned **DistilBERT** model to categorize user issues across 27 unique IT intents (e.g., password recovery, refund policy, account issues).
* **Semantic Retrieval:** Utilizes **FAISS** (Facebook AI Similarity Search) and **Sentence-Transformers** to retrieve the most relevant enterprise policy from a local knowledge base.
* **Contextual Generation:** Implements the highly efficient **Qwen2.5-0.5B** generative model to synthesize the retrieved policy into a professional, human-like response.
* **Optimized for CPU:** Engineered to run inference efficiently on standard cloud CPU environments without requiring massive GPU compute clusters.

##  Tech Stack
* **Language:** Python
* **ML Frameworks:** PyTorch, Hugging Face Transformers, Sentence-Transformers
* **Vector DB:** FAISS
* **Web Interface:** Streamlit
* **Models:** DistilBERT (Fine-tuned), Qwen2.5-0.5B (Generative)

##  Project Structure
* `streamlit_app.py`: The production-ready Streamlit web application logic.
* `AutoResolve_NLP_Shaury_Singh.ipynb`: The complete training pipeline, including dataset preprocessing, mathematical baselines (TF-IDF/SVM), and DistilBERT fine-tuning.
* `Shaury_Final_NLP.pdf`: The comprehensive academic report detailing the project's methodology, architecture, and qualitative error analysis.
* `requirements.txt`: Environment dependencies.

##  How It Works
1. **Classify:** The user input is sent to the fine-tuned DistilBERT model to detect the specific intent.
2. **Retrieve:** Based on the query, the system performs a vector search against the internal Knowledge Base using FAISS.
3. **Resolve:** The Generative LLM takes the original query + the retrieved document to generate a precise, hallucination-free solution.

##  Installation & Usage
To run this application locally on your machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/shaury05/AutoResolve-NLP-Triage-RAG-Support-Agent.git](https://github.com/shaury05/AutoResolve-NLP-Triage-RAG-Support-Agent.git)
   cd AutoResolve-NLP-Triage-RAG-Support-Agent
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
Developed by Shaury Pratap Singh as part of the Master of Science in Data Science program at the New Jersey Institute of Technology (NJIT).
