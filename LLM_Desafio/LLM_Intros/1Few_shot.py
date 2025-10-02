import ollama

def chat(messages):
    r = ollama.chat(
    model="llama3",
    messages=messages)

    return r['message']['content']

system = "You are a cibersecurity expert. Answer the user's questions in a concise manner."

# --- Few-shot ---
Few_shot = [
    {"role": "system", "content": system},
    {"role": "user", "content": "What is a firewall?"},
    {"role": "assistant", "content": "A firewall is a network security device that monitors and controls incoming and outgoing network traffic based on predetermined security rules."},
    {"role": "user", "content": "What is an apple?"},
    {"role": "assistant", "content": "Sorry, but the topic does not seem to be related to cybersecurity, try to be more specific."},
    {"role": "user", "content": "What is a VPN?"},
    {"role": "assistant", "content": "A VPN, or Virtual Private Network, is a service that creates a secure and encrypted connection over a less secure network, such as the internet. It helps protect your online privacy and data by masking your IP address and encrypting your internet traffic."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI, or Artificial Intelligence, can be used in cybersecurity to enhance threat detection, automate responses to security incidents, and improve overall system defenses by analyzing large volumes of data for patterns and anomalies."},
        # --- Question to answer ---       
    {"role": "user", "content": "O que é a CIA tria?"},
    {"role": "user", "content": "O que é a framework STRIDE?"},
    {"role": "user", "content": "O que é Data Science?"}
]

print("--- Few-shot ---")
print(chat(Few_shot), end="\n\n")


