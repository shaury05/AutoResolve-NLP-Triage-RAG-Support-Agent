import streamlit as st
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle

st.set_page_config(page_title="AutoResolve Agent", page_icon="🤖", layout="centered")

st.title("🤖 AutoResolve: IT Support Agent")
st.markdown("This end-to-end LLM Agent classifies your IT issue, retrieves the relevant enterprise policy, and generates a solution.")

# --- 1. Load Models (Cached so they only load once) ---
@st.cache_resource
def load_pipeline():
    # 1. Load DistilBERT Classifier from your HF Model Repo
    # 1. Load DistilBERT Classifier from your HF Model Repo
    repo_id = "Shauryaaa05/AutoResolve-DistilBERT"
    folder_name = "autoresolve_distilbert_final"
    
    distil_tokenizer = DistilBertTokenizerFast.from_pretrained(repo_id, subfolder=folder_name)
    distil_model = DistilBertForSequenceClassification.from_pretrained(repo_id, subfolder=folder_name)

    # 2. Knowledge Base & Retriever
    kb = [
        "Refund Policy: Customers are entitled to a full refund within 30 days of purchase. To process, verify the order number and issue the refund to the original payment method.",
        "Order Tracking: To locate an order, query the shipping database using the 10-digit order number. If the status is 'Dispatched', provide the user with the carrier tracking link.",
        "Password Recovery: If a user cannot log in, send a secure password reset link to their registered email address. Ensure they check their spam folder.",
        "Payment Issues: If a transfer or payment fails, verify if the credit card is expired or if the anti-fraud system flagged the transaction. Recommend trying a different payment method."
    ]
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = embedder.encode(kb, convert_to_numpy=True)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
    
    # 3. Load Generative LLM (CPU mode)
    llama_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    llama_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="cpu")
    
    return distil_tokenizer, distil_model, kb, embedder, index, llama_tokenizer, llama_model

with st.spinner("Loading AI Models... (This takes about 60 seconds on initial boot)"):
    distil_tokenizer, distil_model, knowledge_base, embedder, index, llama_tokenizer, llama_model = load_pipeline()

# Define the intents manually to avoid needing the full dataset for the LabelEncoder
INTENTS = ['cancel_order', 'change_order', 'change_shipping_address', 'check_cancellation_fee', 'check_invoice', 'check_payment_methods', 'check_refund_policy', 'complaint', 'contact_customer_service', 'contact_human_agent', 'create_account', 'delete_account', 'delivery_options', 'delivery_period', 'edit_account', 'get_invoice', 'get_refund', 'newsletter_subscription', 'payment_issue', 'place_order', 'recover_password', 'registration_problems', 'review', 'set_up_shipping_address', 'switch_account', 'track_order', 'track_refund']

# --- 2. The User Interface ---
user_query = st.text_input("Describe your IT or Support issue:", placeholder="e.g., am I entitled to a reimbursement?")

if st.button("Submit Ticket"):
    if user_query:
        with st.spinner("Processing..."):
            # Step A: Intent Classification
            inputs = distil_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = distil_model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            predicted_intent = INTENTS[predicted_class_id]
            
            st.success(f"**Intent Classified:** `{predicted_intent}`")
            
            # Step B: Retrieval
            query_vector = embedder.encode([user_query], convert_to_numpy=True)
            distances, indices = index.search(query_vector, 1)
            retrieved_doc = knowledge_base[indices[0][0]]
            
            st.info(f"**Retrieved Knowledge Base Document:** {retrieved_doc}")
            
            # Step C: Generation
            prompt = f"""<|im_start|>system
You are AutoResolve, an IT support agent. Answer the user's query using ONLY the provided IT Document. Be polite, concise, and professional.<|im_end|>
<|im_start|>user
User Query: {user_query}
IT Document: {retrieved_doc}<|im_end|>
<|im_start|>assistant
"""
            gen_inputs = llama_tokenizer(prompt, return_tensors="pt")
            outputs = llama_model.generate(**gen_inputs, max_new_tokens=150, temperature=0.1, pad_token_id=llama_tokenizer.eos_token_id)
            
            full_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_answer = full_response.split("assistant\n")[-1].strip()
            
            st.write("### 💬 AutoResolve Agent Response:")
            st.write(f"> {final_answer}")
    else:
        st.warning("Please enter a query first.")