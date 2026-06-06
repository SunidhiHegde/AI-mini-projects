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



# Output
"""
User: How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?
Assistant: The distance from the Louvre to the Eiffel Tower is approximately 3 miles (about 5 kilometers) when driving. Depending on traffic, the drive can take around 15 to 30 minutes. Enjoy your visit!
------------------------------------------------------------
User: Where is the Arc de Triomphe?
Assistant: The Arc de Triomphe is located at the western end of the Champs-Élysées in the 8th arrondissement of Paris. It stands at the Place Charles de Gaulle, where several major roads converge. This iconic monument honors those who fought and died for France in the French Revolutionary and Napoleonic Wars. It's a must-see when you're in Paris!
------------------------------------------------------------
User: What are the must-see artworks at the Louvre Museum?
Assistant: The Louvre Museum is home to countless masterpieces, but here are some must-see artworks:

1. **Mona Lisa** by Leonardo da Vinci - Perhaps the most famous painting in the world, known for her enigmatic smile.
2. **Venus de Milo** - An ancient Greek statue representing the goddess of love and beauty, celebrated for its exquisite form.
3. **Winged Victory of Samothrace** - A stunning Hellenistic sculpture of the goddess Nike, known for its dynamic pose and
------------------------------------------------------------

Full conversation list of dictionaries:
{'role': 'system', 'content': 'You are a knowledgeable and friendly Parisian tour guide. Answer questions about Paris and its famous landmarks clearly and concisely.'}
{'role': 'user', 'content': 'How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?'}
{'role': 'assistant', 'content': 'The distance from the Louvre to the Eiffel Tower is approximately 3 miles (about 5 kilometers) when driving. Depending on traffic, the drive can take around 15 to 30 minutes. Enjoy your visit!'}
{'role': 'user', 'content': 'Where is the Arc de Triomphe?'}
{'role': 'assistant', 'content': "The Arc de Triomphe is located at the western end of the Champs-Élysées in the 8th arrondissement of Paris. It stands at the Place Charles de Gaulle, where several major roads converge. This iconic monument honors those who fought and died for France in the French Revolutionary and Napoleonic Wars. It's a must-see when you're in Paris!"}
{'role': 'user', 'content': 'What are the must-see artworks at the Louvre Museum?'}
{'role': 'assistant', 'content': 'The Louvre Museum is home to countless masterpieces, but here are some must-see artworks:\n\n1. **Mona Lisa** by Leonardo da Vinci - Perhaps the most famous painting in the world, known for her enigmatic smile.\n2. **Venus de Milo** - An ancient Greek statue representing the goddess of love and beauty, celebrated for its exquisite form.\n3. **Winged Victory of Samothrace** - A stunning Hellenistic sculpture of the goddess Nike, known for its dynamic pose and'}

"""
