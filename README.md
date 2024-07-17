# Chatbot with Short-term and Long-term Memory and RAG

This FastAPI-based Python script implements a chatbot with short-term and long-term memory capabilities using Retrieval-Augmented Generation (RAG). The system maintains context across conversations and can retrieve relevant information from past interactions.

## Key Features

1. **Short-term Memory**: Stores recent messages for immediate context.
2. **Long-term Memory**: Creates summaries of conversation chunks, focusing on places and important events.
3. **Retrieval-Augmented Generation (RAG)**: Uses embeddings to find relevant past conversation summaries.
4. **Simulated Conversations**: Can generate conversations with or without LLM for testing and initialization.

## How It Works

1. **Message Processing**:

   - Each incoming message is added to short-term memory.
   - Long-term memory summaries are created when enough messages accumulate.

2. **Long-term Memory Creation**:

   - Summarizes chunks of conversation.
   - Checks for mentions of places or events.
   - Generates embeddings for efficient retrieval.

3. **Context Enrichment**:

   - Includes recent messages from short-term memory.
   - Retrieves relevant summaries from long-term memory based on similarity.

4. **Simulated Conversations**:
   - Can generate conversations using GPT-4 or a rule-based approach.
   - Used for initializing the system with sample data.

## API Endpoints

- `/enrich_message`: Processes a new message and returns enriched context.
- `/finalize_session/{user_id}`: Finalizes a session, ensuring all messages are processed into long-term memory.
- `/memory_state/{user_id}`: Retrieves the current state of the memory for a user.

## Setup and Usage

1. Set up environment variables (OpenAI API key).
2. Run the script to start the FastAPI server.
3. The system initializes with a simulated conversation if no existing memory is found.
4. Use the API endpoints to interact with the chatbot and manage its memory.

## Dependencies

- FastAPI
- OpenAI API
- scikit-learn (for cosine similarity)
- dotenv (for environment variables)

This system demonstrates an approach to maintaining context in chatbot conversations, allowing for more coherent and informative interactions over time.
