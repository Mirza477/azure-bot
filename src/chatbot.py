# src/chatbot.py
import openai
from config.settings import OPENAI_COMPLETIONS_DEPLOYMENT
from src.cosmos_db import query_vector_search
from src.embeddings import generate_embedding

# Global conversation history for the session.
conversation_history = []

def summarize_history(history):
    """
    Summarizes earlier parts of the conversation succinctly.
    """
    prompt = "Summarize the following conversation succinctly, capturing only key points:\n"
    for msg in history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=400  # Limit summary length
        )
        summary = response["choices"][0]["message"]["content"].strip()
        print("Summary generated:", summary, flush=True)
        return summary
    except Exception as e:
        print("Error in summarize_history:", e, flush=True)
        return ""

def generate_response(user_query: str):
    global conversation_history
    print("generate_response called with:", user_query, flush=True)
    
    # Append current user query to conversation history.
    conversation_history.append({"role": "user", "content": user_query})
    
    # For retrieval, use only the current query to avoid contamination from previous queries.
    retrieval_text = user_query
    
    try:
        query_embedding = generate_embedding(retrieval_text)
        print("Embedding generated", flush=True)
    except Exception as e:
        error_msg = f"Error generating embedding: {e}"
        print(error_msg, flush=True)
        conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

    try:
        # Retrieve top 3 relevant documents to provide a richer context.
        docs = query_vector_search(query_embedding, top_k=10)
        print("Relevant docs:", docs, flush=True)
    except Exception as e:
        error_msg = f"Error querying Cosmos DB: {e}"
        print(error_msg, flush=True)
        conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

    # # Build an enhanced system prompt.
#     system_prompt = (
#     "You are an AKU Enterprise Employee Support Assistant designed to help employees and students by answering questions across multiple departments, including: "
#     "ICT (Information & Communication Technology), Human Resources (HR), Finance & Payroll, Facilities Management, Procurement & Supply Chain, "
#     "Infection Control, Academic Services, Clinical Administration, and more. Use only the official knowledgebase and policy documents provided by these departments. "
#     "Do not use any external sources or your own background knowledge. Respond accurately, concisely, and professionally — as if you are a trained helpdesk representative from the relevant department. "
#     "Always reflect the latest AKU policies and procedures. If a topic is governed by a formal SOP, policy, or form (e.g., leave policy, infection protocol, procurement process), mention it clearly. "
#     "When guiding users through tasks (e.g., submitting a leave request, raising a PR, requesting room repair), break down steps in numbered points for clarity. "
#     "If the answer cannot be found in the provided documents, respond with: "
#     "'I’m not trained for this. Please contact the relevant department or your supervisor.' "
#     "Avoid small talk and greetings — be polite but focused on helping quickly. Don’t reference yourself as an AI unless directly asked. "
#     "If a user’s question is vague, ask for more details to provide the most accurate help."
# )


    system_prompt = (

        "You are an expert AI assistant on company policies. "
        "Use the document excerpts provided below along with the current user query to generate a detailed and clear answer. "
        "If the query is ambiguous or lacks sufficient details, ask clarifying questions instead of guessing. "
        "Answer in 2-3 sentences with precise policy details."
        "Never answer in abusive tone or harsh tone even the user is using one. "
        "If the query is not directly about the company's policies or the content in the provided documents, respond with: "
        "'I am not trained for this. Please ask relevant questions'"

        # "You are an expert AI assistant on company internal information. Use the document excerpts provided below along with the current user query to generate a detailed and clear answer. If the query is ambiguous or lacks sufficient details, ask clarifying questions. For follow-up questions that request further explanation on a topic already introduced, please expand on the details of the relevant policy in a clear manner, rather than defaulting to 'I am not trained for this."
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    # Process each retrieved document: include a brief excerpt (first 200 characters) for context.
    if docs:
        for doc in docs:
            doc_content = doc.get('content', '')
            brief_content = doc_content[:200]  # Only take an excerpt of 200 characters.
            doc_context = (
                f"Document: {doc.get('document_name', 'N/A')}, Section: {doc.get('section', 'N/A')}.\n"
                f"Excerpt: {brief_content}"
            )
            messages.append({"role": "system", "content": doc_context})
    else:
        messages.append({"role": "system", "content": "No relevant documents found."})
    
    # Optionally: include recent conversation context (if you believe context aids clarity).
    # If you want to include some conversation context, you could add, for example, the last 2 exchanges.
    if len(conversation_history) > 2:
        recent_context = conversation_history[-2:]
        messages.extend(recent_context)
    else:
        messages.append({"role": "user", "content": user_query})
    
    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.2,
            max_tokens=250  # Limit answer length
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print("Answer received:", answer, flush=True)
    except Exception as e:
        answer = f"Error in ChatCompletion: {e}"
        print(answer, flush=True)
    
    conversation_history.append({"role": "assistant", "content": answer})
# Return a dictionary with the answer and an empty list for results
    return {"response": answer, "results": []}

