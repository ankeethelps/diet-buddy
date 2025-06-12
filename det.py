import streamlit as st
import base64
import re
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Groq client
groq_client = Groq(api_key="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")

# Chatbot model
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_client.api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# System prompt for the AI dietitian
SYSTEM_PROMPT = """
You are a Great dietitian who has a great knowledge of food and its nutrients. Answer the user's question clearly.
If an image is provided, first describe what it is, for example, "INDIAN THALI".
Then, answer the question based on it. Imagine it is a veg thali, then your response should be clear and straightforward in the format:
write the item its  nutrients alongside no need to write the item name again , and list them vertically 
[Food_Item_1:Xcalories,
Food_Item_2:Ycalories], 
Total Calories: Z kcal, Protein: P g, Sugar: S g.
(Example: [Rice:100calories, 
Dal:200calories], 
Total Calories: 300 kcal, Protein: 15 g, Sugar: 5 g)

If a user sends you a product photo, then you can directly show the nutrition in the format:
Calories: X kcal, Protein: P g, Sugar: S g.
(Example: Calories: 250 kcal, Protein: 10 g, Sugar: 20 g)

After providing the nutrition, you can add a short sentence in "hinglish" as "THE AMOUNT OF WORKOUT OR THE DURATION OF THE WORKOUT
TO BURNOUT THE CALORIES e.g., "YOU NEED TO RUN X KILOMETRE , OR YOU NEED TO SPRINT FOR X MINUTES " suggest different workout plan !").
If no image is provided, answer the question using your general knowledge or search it on the web using tools.
"""

def parse_nutrition(text):
    """Parses nutrition data (calories, protein, sugar) from the LLM's response text."""
    calories = 0
    protein = 0.0
    sugar = 0.0

    # Regex to find calories, prioritizing "Total Calories" then "Calories"
    calories_match = re.search(r"(?:Total\s*)?Calories:\s*(\d+)\s*kcal", text, re.IGNORECASE)
    if calories_match:
        calories = int(calories_match.group(1))

    # Regex to find protein in grams
    protein_match = re.search(r"Protein:\s*(\d+(?:\.\d+)?)\s*g", text, re.IGNORECASE)
    if protein_match:
        protein = float(protein_match.group(1))

    # Regex to find sugar in grams
    sugar_match = re.search(r"Sugar:\s*(\d+(?:\.\d+)?)\s*g", text, re.IGNORECASE)
    if sugar_match:
        sugar = float(sugar_match.group(1))

    return calories, protein, sugar

def process_chat_message(user_prompt, image_for_this_message):
    """
    Processes user input (text and optional image), queries the Groq API via LangChain,
    parses the nutrition information from the AI's response, and accumulates
    the totals in Streamlit's session state.
    """
    if not user_prompt and not image_for_this_message:
        st.error("Please enter a prompt or upload an image.")
        return

    # Prepare content for the HumanMessage (multimodal support)
    user_content_parts = [{"type": "text", "text": user_prompt}]
    
    # Store user's message in chat history immediately, including the actual file object for re-display
    st.session_state.messages.append({"role": "user", "content": user_prompt, "image": image_for_this_message})

    if image_for_this_message:
        # Read image bytes and encode to base64
        img_bytes = image_for_this_message.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        user_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    # Construct messages for LangChain's LLM
    messages_for_llm = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content_parts)
    ]
    try:
        with st.spinner("Analyzing..."):
            # Invoke the LangChain ChatGroq model
            llm_response = llm.invoke(messages_for_llm)
            llm_response_text = llm_response.content

        # Parse nutrition from the AI's response text
        calories, protein, sugar = parse_nutrition(llm_response_text)

        # Update the accumulated totals in Streamlit's session state
        st.session_state["total_calories"] += calories
        st.session_state["total_protein"] += protein
        st.session_state["total_sugar"] += sugar

        # Format the AI's full response for display in the chat, including accumulated totals
        formatted_ai_response = (
            f"{llm_response_text}\n\n"
            f"--- Accumulated Nutrition ---\n"
            f"Calories: {st.session_state['total_calories']} kcal\n"
            f"Protein: {st.session_state['total_protein']} g\n"
            f"Sugar: {st.session_state['total_sugar']} g"
        )
        # Add AI's response to chat history
        st.session_state.messages.append({"role": "ai", "content": formatted_ai_response})

    except Exception as e:
        error_message = f"Error processing request: {e}"
        st.error(error_message)
        # Add error message to chat history for user visibility
        st.session_state.messages.append({"role": "ai", "content": error_message})

def reset_chat_and_nutrition():
    """Resets the entire chat history and accumulated nutrition data."""
    st.session_state["total_calories"] = 0
    st.session_state["total_protein"] = 0.0
    st.session_state["total_sugar"] = 0.0
    st.session_state["messages"] = [] # Clear the chat messages
    
    # Crucial for clearing file uploader display
    st.session_state["file_uploader_key"] += 1 
    st.session_state["current_uploaded_file"] = None # Also clear the stored file object

    st.success("Chat and accumulated nutrition data have been reset!")


# --- Streamlit App Layout ---
st.set_page_config(page_title="Groq Vision Dietitian Chat", layout="centered")

st.title("YOUR DIETECIAN Chat")
st.markdown("Upload a food image. Your personalized dietitian will respond!")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_calories" not in st.session_state:
    st.session_state["total_calories"] = 0
if "total_protein" not in st.session_state:
    st.session_state["total_protein"] = 0.0
if "total_sugar" not in st.session_state:
    st.session_state["total_sugar"] = 0.0
# Key for the file uploader to manage its state and force reset
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
# Variable to hold the actual uploaded file object that's ready for processing
if "current_uploaded_file" not in st.session_state:
    st.session_state["current_uploaded_file"] = None


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # If it's a user message and an image was associated, display it
        if message["role"] == "user" and message["image"]:
            st.image(message["image"], caption="Your Upload", width=200)

# Input for new messages and image upload (placed in sidebar)
with st.sidebar:
    # File uploader, using the dynamic key
    uploaded_file_from_sidebar = st.file_uploader(
        "Upload Food Image (Optional)",
        type=["jpg", "jpeg", "png"],
        key=st.session_state["file_uploader_key"]
    )
    
    # If a new file is uploaded through the sidebar, store it in session_state.current_uploaded_file.
    # This also handles the case where a user removes an uploaded file from the uploader,
    # as `uploaded_file_from_sidebar` would become None.
    if uploaded_file_from_sidebar is not None and uploaded_file_from_sidebar != st.session_state["current_uploaded_file"]:
        st.session_state["current_uploaded_file"] = uploaded_file_from_sidebar
        st.write("Image uploaded! Now type your question in the chat below to analyze it.")
        st.experimental_rerun() # Rerun to update sidebar display and prompt user for text

    st.markdown("---")
    st.subheader("Current Accumulated Nutrition")
    st.write(f"**Calories:** {st.session_state['total_calories']} kcal")
    st.write(f"**Protein:** {st.session_state['total_protein']} g")
    st.write(f"**Sugar:** {st.session_state['total_sugar']} g")
    st.markdown("---")
    if st.button("Reset Chat & Nutrition", type="secondary", use_container_width=True):
        reset_chat_and_nutrition()
        st.experimental_rerun() # Rerun to clear the display immediately

# Chat input at the bottom of the main page
user_prompt = st.chat_input("Ask about your food or nutrition...")

# This is the crucial part:
# Process the new message if the user has typed something in the chat input
if user_prompt:
    # Get the image that's currently 'staged' for the next message from session state.
    image_to_process = st.session_state["current_uploaded_file"]

    # Call the processing function with the text prompt and the staged image (if any)
    process_chat_message(user_prompt, image_to_process)

    # After processing this message, clear the staged image from session_state
    # This prevents the image from being re-used for subsequent text-only prompts.
    st.session_state["current_uploaded_file"] = None
    
    # Increment the file uploader key to visually clear the uploaded file in the sidebar.
    st.session_state["file_uploader_key"] += 1 
    
    st.experimental_rerun() # Force a rerun to update chat history and clear uploader.
