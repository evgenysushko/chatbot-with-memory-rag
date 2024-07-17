import os
import json
import logging
import random
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Constants
LONG_TERM_MEMORY_WINDOW = 50
# LONG_TERM_MEMORY_STEP = 10 # In case we want to implement a sliding window 
SIMILARITY_THRESHOLD = 0.7
MAX_RELEVANT_SUMMARIES = 3
SIMULATED_CONVERSATION_LENGTH = 100
SHORT_TERM_CONTEXT_LENGTH = 10
MEMORY_FILE_PATTERN = "memory_{}.json"

class Message(BaseModel):
    content: str
    user_id: str

class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term_memory: List[str] = []
        self.long_term_memory: List[Dict] = []
        self.message_count = 0
        self.last_long_term_memory_index = 0

    def add_to_short_term_memory(self, message: str) -> None:
        self.short_term_memory.append(message)
        self.message_count += 1

    def add_to_long_term_memory(self) -> None:
        if self.message_count > self.last_long_term_memory_index:
            start_index = self.last_long_term_memory_index
            end_index = min(start_index + LONG_TERM_MEMORY_WINDOW, self.message_count)
            messages_to_summarize = self.short_term_memory[start_index:end_index]
            
            if messages_to_summarize:
                summary = create_summary(messages_to_summarize)
                embedding = generate_embedding(summary)
                places_events = check_places_and_events(messages_to_summarize)
                
                self.long_term_memory.append({
                    "summary": summary,
                    "embedding": embedding,
                    "places_events": places_events,
                    "original_messages": messages_to_summarize,
                    "start_index": start_index,
                    "end_index": end_index - 1
                })
                
                self.last_long_term_memory_index = end_index

    def process_message(self, message: str) -> None:
        if not message.startswith("User: ") and not message.startswith("Assistant: "):
            message = f"User: {message}"
        
        self.add_to_short_term_memory(message)
        
        # Check if we need to create a new long-term memory
        if self.message_count - self.last_long_term_memory_index >= LONG_TERM_MEMORY_WINDOW:
            self.add_to_long_term_memory()
        
        self.save_to_file()

    def finalize_session(self) -> None:
        # Create final long-term memory if there are remaining messages
        if self.message_count > self.last_long_term_memory_index:
            self.add_to_long_term_memory()
        self.save_to_file()

    def create_long_term_memories(self) -> None:
        while self.last_long_term_memory_index < self.message_count:
            self.add_to_long_term_memory()


    def get_relevant_summaries(self, query: str) -> List[Dict]:
        query_embedding = generate_embedding(query)
        relevant_summaries = []

        for memory in self.long_term_memory:
            if memory['places_events'] == 1:
                similarity = cosine_similarity([query_embedding], [memory['embedding']])[0][0]
                if similarity > SIMILARITY_THRESHOLD:
                    relevant_summaries.append({
                        'summary': memory['summary'],
                        'similarity': similarity
                    })

        relevant_summaries.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_summaries[:MAX_RELEVANT_SUMMARIES]

    def get_memory_state(self) -> Dict:
        return {
            "user_id": self.user_id,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
            "message_count": self.message_count,
            "last_long_term_memory_index": self.last_long_term_memory_index
        }

    def save_to_file(self) -> None:
        filename = MEMORY_FILE_PATTERN.format(self.user_id)
        with open(filename, 'w') as f:
            json.dump(self.get_memory_state(), f)

    @classmethod
    def load_from_file(cls, user_id: str) -> 'MemoryManager':
        filename = MEMORY_FILE_PATTERN.format(user_id)
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            manager = cls(user_id)
            manager.short_term_memory = data['short_term_memory']
            manager.long_term_memory = data['long_term_memory']
            manager.message_count = data['message_count']
            manager.last_long_term_memory_index = data.get('last_long_term_memory_index', 0)
            return manager
        except FileNotFoundError:
            return cls(user_id)

def get_memory_manager(user_id: str) -> MemoryManager:
    return MemoryManager.load_from_file(user_id)

def create_summary(messages: List[str]) -> str:
    prompt = f"Summarize the following conversation, focusing on key information, especially mentioned places and important events:\n\n"
    prompt += "\n".join(messages)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def check_places_and_events(messages: List[str]) -> int:
    prompt = f"Analyze the following conversation and determine if any specific places or important events are mentioned. Output only the number 1 if places or events are found, or 0 otherwise:\n\n"
    prompt += "\n".join(messages)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations for places and events. Respond only with 0 or 1."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content.strip()
    
    # Extract the number from the result
    try:
        return int(result.split()[-1])
    except ValueError:
        logger.warning(f"Unexpected response from check_places_and_events: {result}")
        return 0  # Default to 0 if we can't parse the response

def generate_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def check_if_question_about_place_or_event(message: str) -> bool:
    prompt = f"Determine if the following message is asking about a place or an important event that was likely mentioned previously in the conversation. Output 'yes' if it is, and 'no' otherwise:\n\n{message}"
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes messages."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content.strip().lower()
    return result == 'yes'


def generate_simulated_conversation() -> List[str]:
    place_indices = random.sample(range(0, SIMULATED_CONVERSATION_LENGTH // 2), 2)
    event_indices = random.sample([i for i in range(0, SIMULATED_CONVERSATION_LENGTH // 2) if i not in place_indices], 2)
    
    prompt = f"""Generate a random conversation between a user and an AI assistant with exactly {SIMULATED_CONVERSATION_LENGTH} messages, alternating between user and assistant.
    Each next message should be random and doesn't depend on the previous.
    In the user's messages at indices {place_indices}, mention a specific place.
    In the user's messages at indices {event_indices}, mention an important event.
    All other user messages should not mention any specific places or events.
    Format each message as 'User: [message]' or 'Assistant: [message]'.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        content = response.choices[0].message.content.strip()
        messages = [msg.strip() for msg in content.split('\n') if msg.strip()]                
        
        logger.info(f"Successfully generated a conversation with {len(messages)} messages")
        return messages
    
    except Exception as e:
        logger.error(f"Error in generate_simulated_conversation: {str(e)}")
        raise

def generate_simulated_conversation_without_llm() -> List[str]:
    half_conv_length = SIMULATED_CONVERSATION_LENGTH
    conversation = []
    places = ["New York", "Paris", "Tokyo", "London", "Sydney", "Moscow", "Cairo", "Rio de Janeiro"]
    events = ["World War II", "Moon Landing", "Fall of the Berlin Wall", "9/11 Attacks", "COVID-19 Pandemic", "Industrial Revolution"]
    
    place_indices = random.sample(range(0, half_conv_length), 2)
    event_indices = random.sample([i for i in range(half_conv_length) if i not in place_indices], 2)
    
    for i in range(half_conv_length):
        # User message
        if i in place_indices:
            place = random.choice(places)
            user_message = f"User: Have you ever been to {place}? I heard it's beautiful."
        elif i in event_indices:
            event = random.choice(events)
            user_message = f"User: What do you know about {event}? It was a significant historical event."
        else:
            user_message = f"User: This is message number {i*2+1} from the user."
        conversation.append(user_message)
        
        # Assistant message
        assistant_message = f"Assistant: This is response number {i*2+2} from the assistant."
        conversation.append(assistant_message)
    
    return conversation

def initialize_long_term_memory(use_llm: bool = True):
    logger.info("Generating simulated conversation")
    try:
        if use_llm:
            conversation = generate_simulated_conversation()
        else:
            conversation = generate_simulated_conversation_without_llm()
        
        memory_manager = MemoryManager("001")
        
        for message in conversation:
            memory_manager.process_message(message)
        
        memory_manager.create_long_term_memories()
        
        logger.info(f"Long-term memory initialized with {memory_manager.message_count} messages")
        logger.info(f"Created {len(memory_manager.long_term_memory)} long-term memory entries")
        for i, entry in enumerate(memory_manager.long_term_memory):
            logger.info(f"Entry {i}: indices {entry['start_index']} to {entry['end_index']}")
        
        memory_manager.save_to_file()
        print("Initialized long-term memory with simulated conversation.")
    except Exception as e:
        logger.error(f"Error initializing long-term memory: {str(e)}")
        raise

@app.post("/enrich_message")
async def enrich_message(message: Message, memory_manager: MemoryManager = Depends(get_memory_manager)):
    # Process and save the user message
    memory_manager.process_message(message.content)
    
    # Get the last N messages from short-term memory
    recent_messages = memory_manager.short_term_memory[-SHORT_TERM_CONTEXT_LENGTH:]
    
    enriched_context = "Recent conversation:\n"
    enriched_context += "\n".join(recent_messages)
    enriched_context += f"\n\nUser message: {message.content}"
    
    if check_if_question_about_place_or_event(message.content):
        relevant_summaries = memory_manager.get_relevant_summaries(message.content)
        if relevant_summaries:
            enriched_context += "\n\nRelevant information from previous conversations:"
            for summary in relevant_summaries:
                enriched_context += f"\n- {summary['summary']}"
    
    return {"enriched_context": enriched_context}

@app.post("/finalize_session/{user_id}")
async def finalize_session(user_id: str, memory_manager: MemoryManager = Depends(get_memory_manager)):
    memory_manager.finalize_session()
    return {"message": "Session finalized and long-term memories created"}

@app.get("/memory_state/{user_id}")
async def get_memory_state(user_id: str, memory_manager: MemoryManager = Depends(get_memory_manager)):
    return memory_manager.get_memory_state()

if __name__ == "__main__":    
    # Initialize long-term memory if it doesn't exist
    if not os.path.exists(MEMORY_FILE_PATTERN.format("001")):
        initialize_long_term_memory(use_llm=False)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
