# AutoResolve: Enterprise RAG Agent for IT Support 🤖

AutoResolve is a sophisticated AI Agent designed to completely automate enterprise IT support ticket triage and resolution. It combines custom Intent Classification, Retrieval-Augmented Generation (RAG), and a Generative LLM to provide instant, policy-accurate solutions to user queries.

### 🔗 [Live Web App Demo on Hugging Face Spaces](https://huggingface.co/spaces/Shauryaaa05/AutoResolve-Agent)

## 💡 The Inspiration
During my time as an Analyst Programmer at **Ramco Systems** (Aviation Business Unit), I witnessed a massive operational bottleneck: managers were spending hours manually reading support tickets and routing them to specific departments (like the execution or flight safety teams). I realized this manual triage was inefficient, unscalable, and costly. 

I built **AutoResolve** to completely eliminate this manual bottleneck. Instead of a human reading and routing the ticket, AutoResolve automatically classifies the issue, retrieves the correct enterprise policy, and instantly generates a solution for the user.

## 🌟 Key Features
* **Intent Classification:** Uses a fine-tuned **DistilBERT** model to categorize messy, unstructured user issues across 27 unique IT intents with **99.81% accuracy**.
* **Semantic Retrieval (RAG):** Utilizes **FAISS** (Facebook AI Similarity Search) and **Sentence-Transformers** to retrieve the most relevant enterprise policy from a local knowledge base.
* **Contextual Generation:** Implements the highly efficient **Qwen2.5-0.5B** generative model to synthesize the retrieved policy into a professional, hallucination-free response.
* **Optimized for CPU Deployment:** Engineered to run inference efficiently on standard cloud CPU environments without requiring massive, expensive GPU compute clusters.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Frameworks:** PyTorch, Hugging Face Transformers, Sentence-Transformers, scikit-learn
* **Vector DB:** FAISS
* **Web Interface:** Streamlit
* **Models:** DistilBERT (Fine-tuned for Classification), Qwen2.5-0.5B (Generative LLM)

## 📂 Project Structure
* `streamlit_app.py`: The production-ready Streamlit web application logic.
* `AutoResolve_NLP_Shaury_Singh.ipynb`: The complete training pipeline, including dataset preprocessing (custom Regex masking), mathematical baselines (TF-IDF/SVM), and DistilBERT fine-tuning. *(Note: The GitHub version has cleared outputs to meet file size limits. **[View the full notebook with all training outputs and charts here](https://drive.google.com/file/d/1xKIzkyQuxoRcSgEM0WdYI-jHTUpfSUtQ/view?usp=drive_link)**).*
* `Shaury_Final_NLP.pdf`: The comprehensive academic report detailing the project's methodology, architecture, hardware optimization, and qualitative error analysis.
* `requirements.txt`: Environment dependencies.

## 🚀 How It Works
1. **Classify:** The user input is sent to the fine-tuned DistilBERT model to detect the specific intent (e.g., "password recovery" or "refund policy").
2. **Retrieve:** Based on the query, the system performs a cosine similarity vector search against the internal Knowledge Base using FAISS to pull the factual company rule.
3. **Resolve:** The Generative LLM takes the original query + the retrieved document and is strictly prompted to generate a precise solution using *only* the retrieved context.

## ⚙️ Installation & Local Usage
To run this application locally on your machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/shaury05/AutoResolve-NLP-Triage-RAG-Support-Agent.git](https://github.com/shaury05/AutoResolve-NLP-Triage-RAG-Support-Agent.git)
   cd AutoResolve-NLP-Triage-RAG-Support-Agent
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---
*Developed by **Shaury Pratap Singh** as part of the Master of Science in Data Science program at the New Jersey Institute of Technology (NJIT).*
