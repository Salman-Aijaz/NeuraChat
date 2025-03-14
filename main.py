import ollama
from typing import List, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph

# Define State Schema
class ChatState(BaseModel):
    messages: List[Dict[str, str]] = []  # Stores chat history
    sentiment: str = "neutral"  # User's detected mood
    summary: str = ""  # Summarized conversation
    feedback: str = ""  # User feedback on responses
    user_input: str = ""  # Stores the latest user input

# Sentiment Analysis Function
def analyze_sentiment(message: str) -> str:
    if any(word in message.lower() for word in ["sad", "depressed", "anxious", "upset"]):
        return "negative"
    elif any(word in message.lower() for word in ["happy", "excited", "joyful"]):
        return "positive"
    return "neutral"

# Chatbot Response Function
def chatbot_response(state: ChatState) -> ChatState:
    user_input = state.user_input  # Extract user input from state
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": user_input}])

    new_state = state.model_copy()
    new_state.messages.append({"user": user_input, "bot": response["message"]["content"]})
    new_state.sentiment = analyze_sentiment(user_input)

    return new_state

# Summarization Function
def summarize_conversation(state: ChatState) -> ChatState:
    messages_text = "\n".join([msg["user"] + " " + msg["bot"] for msg in state.messages])
    summary = ollama.chat(model="llama3", messages=[{"role": "user", "content": "Summarize this conversation: " + messages_text}])

    new_state = state.model_copy()
    new_state.summary = summary["message"]["content"]

    return new_state

# ✅ Fix: Ensure `update_state` always returns a `ChatState` instance
def update_state(state: ChatState) -> ChatState:
    new_state = chatbot_response(state)

    if len(new_state.messages) % 5 == 0:  # Summarize after every 5 interactions
        new_state = summarize_conversation(new_state)

    return ChatState(**new_state.model_dump())  # Ensuring correct type

# Define LangGraph Workflow
workflow = StateGraph(ChatState)

workflow.add_node("chatbot", update_state)

# Set Entry Point
workflow.set_entry_point("chatbot")

# Compile the graph
graph = workflow.compile()

# Run Chatbot
if __name__ == "__main__":
    state = ChatState()
    while True:
        user_input = input("You: ")
        state.user_input = user_input  # ✅ Store user input in state
        
        # 🔹 Fix: Ensure `graph.invoke(state)` returns a `ChatState`
        result = graph.invoke(state)
        if isinstance(result, ChatState):
            state = result
        else:
            state = ChatState(**result)

        print(f"Bot: {state.messages[-1]['bot']}")
        print(f"[Mood: {state.sentiment}]\n")
