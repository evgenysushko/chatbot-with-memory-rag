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

1. Install dependencies.
2. Set up environment variables (OpenAI API key).
3. Run the script to start the FastAPI server.
4. The system initializes with a simulated conversation if no existing memory is found.
5. Use the API endpoints to interact with the chatbot and manage its memory.

## Dependencies

- FastAPI
- OpenAI API
- scikit-learn (for cosine similarity)
- dotenv (for environment variables)

## Limitations and Potential Improvements

While this chatbot implementation provides some foundation for context-aware conversations, it has several limitations that could be addressed in future iterations:

1. **Scalability Issues**:

   - **Limitation**: The current in-memory storage of conversations and summaries will not scale well for large numbers of users or very long conversations.
   - **Improvement**: A database solution for storing conversation data, their summaries, embeddings and metadata could be implemented. This would allow for better scalability and persistence.

2. **Limited Context Window**:

   - **Limitation**: The fixed-size windows for short-term and long-term memory may miss important context that falls outside these windows.
   - **Improvement**: A sliding window approach could be implemented, or a more dynamic method of determining what information to keep or summarize.

3. **Simplistic Relevance Determination**:

   - **Limitation**: The current method of determining if a message is about a place or event is relatively simplistic and may miss nuanced references.
   - **Improvement**: More sophisticated NLP techniques could be used, such as named entity recognition or topic modeling, to better understand message content.

4. **Limited Error Handling and Robustness**:

   - **Limitation**: The current implementation may not gracefully handle all types of errors or edge cases.
   - **Improvement**: More comprehensive error handling, input validation, and fallback mechanisms could be implemented to increase system robustness.

5. **Limited Evaluation Metrics**:
   - **Limitation**: The system lacks quantitative measures of its performance and effectiveness.
   - **Improvement**: Evaluation metrics should be used to measure and improve the system's performance over time.
