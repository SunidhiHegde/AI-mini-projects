import os
from openai import OpenAI

# Define the model to use
model = "gpt-4o-mini"

# Define the client
client = OpenAI()
# Initialize conversation with a system message
conversation = [
    {
        "role": "system",
        "content": (
            "You are a knowledgeable and friendly Parisian tour guide. "
            "Answer questions about Paris and its famous landmarks clearly and concisely."
        )
    }
]

# List of tourist questions
questions = [
    "How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?",
    "Where is the Arc de Triomphe?",
    "What are the must-see artworks at the Louvre Museum?"
]

# Send each question and collect responses
for question in questions:
    # Append user question to conversation
    conversation.append({"role": "user", "content": question})

    # Call the OpenAI API with full conversation history
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.0,
        max_tokens=100
    )

    # Extract assistant reply
    assistant_reply = response.choices[0].message.content

    # Append assistant response to conversation
    conversation.append({"role": "assistant", "content": assistant_reply})

    # Print the Q&A
    print(f"User: {question}")
    print(f"Assistant: {assistant_reply}")
    print("-" * 60)

# Display full conversation structure
print("\nFull conversation list of dictionaries:")
for entry in conversation:
    print(entry)
