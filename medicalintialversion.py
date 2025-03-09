import os
from typing import List
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  # Correct import for PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated memory system
from langchain.schema.messages import AIMessage, HumanMessage  # For handling messages

# Set your Groq API key
groqapikey = os.environ.get('api-key')
if not groqapikey:
    print("Error: API key not found. Please set your 'api-key' environment variable.")
    exit()

# Initialize the LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    api_key=groqapikey
)

def parse_medical_documents(file_path: str) -> str:
    """Parse medical documents from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("No text could be extracted from the document.")
        return text
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return "No medical document available."
    except Exception as e:
        print(f"Error while parsing the document: {e}")
        return "No medical document available."

class ReasoningEngine:
    def __init__(self, llm, medical_data: str):
        self.llm = llm
        self.medical_data = medical_data

    def diagnose(self, symptoms: List[str]) -> str:
        """Diagnose based on symptoms and medical knowledge."""
        prompt = f"""
        Based on the following symptoms: {', '.join(symptoms)}, and the medical knowledge extracted, what are the possible diagnoses?
        Provide a concise explanation for each possible diagnosis.

        Medical Knowledge: {self.medical_data[:2000]}  # Truncated to first 2000 characters
        """
        try:
            response = self.llm.invoke(prompt)  # Updated to use `invoke` method for LLMs
            if isinstance(response, AIMessage):
                return response.content  # Extract content from AIMessage object
            return str(response)
        except Exception as e:
            return f"Error during diagnosis: {e}"

# Define Prompt Templates
initial_prompt_template = """
You are a helpful medical assistant. Your goal is to gather information from the patient to help them determine the possible cause of their symptoms. Start by asking the patient what their primary complaint is.

Current conversation:
{history}
Patient: {input}
AI: """

followup_prompt_template = """
You are a helpful medical assistant. You have collected the following information:
{context}

Based on the information, what is the next most relevant question to ask the patient to narrow down potential diagnoses? Be specific and ask only one question. Explain why you are asking this question.

Current conversation:
{history}
Patient: {input}
AI: """

initial_prompt = PromptTemplate(input_variables=["history", "input"], template=initial_prompt_template)
followup_prompt = PromptTemplate(input_variables=["history", "input", "context"], template=followup_prompt_template)

# Initialize Memory (New System)
message_history = ChatMessageHistory()

def save_message_to_history(user_input, ai_response):
    """Save messages to chat history."""
    message_history.add_user_message(user_input)
    message_history.add_ai_message(ai_response)

def get_conversation_history():
    """Retrieve conversation history as a formatted string."""
    messages = message_history.messages
    history = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            history += f"Patient: {message.content}\n"
        elif isinstance(message, AIMessage):
            history += f"AI: {message.content}\n"
    return history

# Create Chains using `|` Operator
initial_chain = initial_prompt | llm
followup_chain = followup_prompt | llm

def main():
    # Load medical data (replace with your actual medical document)
    medical_document_path = "medical_documents.pdf"
    try:
        medical_data = parse_medical_documents(medical_document_path)
    except FileNotFoundError:
        print(f"Error: The file '{medical_document_path}' was not found.")
        medical_data = "No medical document available."

    # Initialize reasoning engine
    engine = ReasoningEngine(llm=llm, medical_data=medical_data)

    print("Welcome to the AI Medical Assistant. Please describe your primary complaint.")

    context = ""
    symptoms = []

    while True:
        try:
            user_input = input("Patient: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting... Thank you for using the AI Medical Assistant!")
                break

            # Get conversation history for context
            context = get_conversation_history()

            if not context:  # First interaction
                ai_response_object = initial_chain.invoke({"history": context, "input": user_input})
            else:
                ai_response_object = followup_chain.invoke({
                    "history": context,
                    "input": user_input,
                    "context": context,
                })

            # Extract AI response content (handle both AIMessage and other formats)
            ai_response_content = (
                ai_response_object.content if isinstance(ai_response_object, AIMessage) else str(ai_response_object)
            )

            print("AI:", ai_response_content)

            # Save messages to chat history
            save_message_to_history(user_input, ai_response_content)
            symptoms.append(user_input)

            # Diagnose after collecting at least 3 symptoms
            if len(symptoms) >= 3:
                print("\nBased on the symptoms provided, here's a potential diagnosis:")
                diagnosis = engine.diagnose(symptoms)
                print(diagnosis)
                print(
                    "\nPlease note: This is not a substitute for professional medical advice. Consult with a healthcare provider for accurate diagnosis and treatment.")

                # Reset symptoms list if you want to continue interacting after diagnosis.
                symptoms.clear()

        except Exception as e:
            print(f"An error occurred during interaction: {e}")

if __name__ == "__main__":
    main()
