"""
Langchain Memory Types - Demonstration

This module showcases different types of memory in Langchain for managing 
conversation history in LLM applications.

Memory Types:
1. ConversationBufferMemory - Stores all messages
2. ConversationBufferWindowMemory - Stores last K messages
3. ConversationSummaryMemory - Summarizes all messages
4. ConversationSummaryBufferMemory - Hybrid of buffer and summary
5. ConversationTokenBufferMemory - Stores messages based on token limit
"""

from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)

print("=" * 80)
print("LANGCHAIN MEMORY TYPES DEMONSTRATION")
print("=" * 80)


# 1. CONVERSATION BUFFER MEMORY

print("\n1. CONVERSATION BUFFER MEMORY")
print("-" * 80)
print("Stores all messages in conversation history")

buffer_memory = ConversationBufferMemory(
    return_messages=True,
    ai_prefix="Assistant"
)

# Create conversation chain
conversation_buffer = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

print("\nUser: What is machine learning?")
response1 = conversation_buffer.invoke({"input": "What is machine learning?"})
print(f"Assistant: {response1['response']}")

print("\nUser: Can you give me examples?")
response2 = conversation_buffer.invoke({"input": "Can you give me examples?"})
print(f"Assistant: {response2['response']}")

print("\nConversation Buffer Memory History:")
print(buffer_memory.buffer)


# 2. CONVERSATION BUFFER WINDOW MEMORY

print("\n\n2. CONVERSATION BUFFER WINDOW MEMORY")
print("-" * 80)
print("Stores only the last K messages (sliding window)")

window_memory = ConversationBufferWindowMemory(
    k=2,  # Keep only last 2 messages
    return_messages=True,
    ai_prefix="Assistant"
)

# Create conversation chain with window memory
conversation_window = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

print("\nAdding multiple messages (only last 2 will be kept)...")

print("\nUser: What is AI?")
conversation_window.invoke({"input": "What is AI?"})

print("\nUser: What is NLP?")
conversation_window.invoke({"input": "What is NLP?"})

print("\nUser: Tell me more about NLP applications")
conversation_window.invoke({"input": "Tell me more about NLP applications"})

print("\nWindow Memory History (Last 2 messages):")
print(window_memory.buffer)


# 3. CONVERSATION SUMMARY MEMORY

print("\n\n3. CONVERSATION SUMMARY MEMORY")
print("-" * 80)
print("Progressively summarizes the conversation history")

summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True,
    ai_prefix="Assistant"
)

# Create conversation chain with summary memory
conversation_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

print("\nUser: I studied computer science")
conversation_summary.invoke({"input": "I studied computer science"})

print("\nUser: I worked for 5 years in tech")
conversation_summary.invoke({"input": "I worked for 5 years in tech"})

print("\nUser: Now I'm interested in AI and machine learning")
conversation_summary.invoke({"input": "Now I'm interested in AI and machine learning"})

print("\nSummary Memory (Summarized History):")
print(summary_memory.buffer)


# 4. CONVERSATION SUMMARY BUFFER MEMORY

print("\n\n4. CONVERSATION SUMMARY BUFFER MEMORY")
print("-" * 80)
print("Hybrid approach - keeps recent messages intact, summarizes older ones")

summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,  # Summarize when exceeding token limit
    return_messages=True,
    ai_prefix="Assistant"
)

# Create conversation chain
conversation_summary_buffer = ConversationChain(
    llm=llm,
    memory=summary_buffer_memory,
    verbose=True
)

messages = [
    "My name is John and I work as a software engineer",
    "I have 8 years of experience in full-stack development",
    "I specialize in Python, JavaScript, and cloud technologies",
    "Recently I've been learning about LangChain and AI applications"
]

print("\nAdding multiple messages...")
for msg in messages:
    print(f"\nUser: {msg}")
    conversation_summary_buffer.invoke({"input": msg})

print("\nSummary Buffer Memory (Recent + Summarized):")
print(summary_buffer_memory.buffer)


# 5. CONVERSATION TOKEN BUFFER MEMORY

print("\n\n5. CONVERSATION TOKEN BUFFER MEMORY")
print("-" * 80)
print("Stores messages based on token limit (not message count)")

token_buffer_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=300,  # Keep up to 300 tokens
    return_messages=True,
    ai_prefix="Assistant"
)

# Create conversation chain
conversation_token_buffer = ConversationChain(
    llm=llm,
    memory=token_buffer_memory,
    verbose=True
)

messages = [
    "I love programming in Python",
    "Python is great for data science and machine learning",
    "LangChain makes it easy to build LLM applications",
    "Memory management is crucial for long conversations"
]

print("\nAdding messages with token limit of 300...")
for msg in messages:
    print(f"\nUser: {msg}")
    conversation_token_buffer.invoke({"input": msg})

print("\nToken Buffer Memory (Token-limited):")
print(token_buffer_memory.buffer)