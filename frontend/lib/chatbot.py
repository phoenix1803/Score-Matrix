import google.generativeai as genai
import sys

genai.configure(api_key="AIzaSyAQvW-7i3jnNu5qwolDOPV9q2HhdkKtrAU")

model = genai.GenerativeModel("gemini-1.5-flash")

conversations = {}

def chat(session_id, user_message):
    """Chat function that maintains history manually."""
    if session_id not in conversations:
        conversations[session_id] = [] 

    history_text = "\n".join([f"User: {msg}" if i % 2 == 0 else f"AI: {msg}" for i, msg in enumerate(conversations[session_id])])
    full_prompt = f"{history_text}\nUser: {user_message}\nAI:"

    response = model.generate_content(full_prompt)

    ai_reply = response.text.strip() if response.text else "I'm not sure."
    conversations[session_id].append(user_message)  
    conversations[session_id].append(ai_reply)  

    return ai_reply

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python chatbot.py <session_id> <user_message>")
        sys.exit(1)

    session_id = sys.argv[1]
    user_message = sys.argv[2]

    try:
        response = chat(session_id, user_message)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
