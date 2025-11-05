import os
import base64
import mimetypes
from typing import List, TypedDict, Literal
from dotenv import load_dotenv
from PIL import Image
import io

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- 1. Define the Graph State ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        input_files: A list of file paths to process.
        parsed_diet: A structured string listing identified foods.
        analysis: A natural language analysis of dietary habits.
        context: Factual info retrieved from the vector store.
        recommendation: The final user-facing recommendation.
    """
    input_files: List[str]
    parsed_diet: str
    analysis: str
    context: str
    recommendation: str

# --- 2. Initialize Models and Retriever ---

# Use a fast model for analysis and generation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Use the powerful vision model for parsing images
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Load the local vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Helper function to process inputs ---

def process_file(file_path):
    """Loads a file and returns its content as a HumanMessage part."""
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type and mime_type.startswith("image/"):
            # It's an image
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
            }
        elif (mime_type and mime_type.startswith("text/")) or not mime_type:
            # Assume text file
            with open(file_path, "r", encoding='utf-8') as text_file:
                return {"type": "text", "text": text_file.read()}
        else:
            print(f"Skipping unsupported file type: {mime_type}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# --- 3. Define the Graph Nodes ---

def parse_input(state: GraphState):
    """
    Node 1: Parse input files (images or text) into a structured food list.
    """
    print("--- 1. PARSING INPUT ---")
    input_files = state["input_files"]
    
    prompt_text = """
    Analyze the provided images and/or text, which represent a user's recent diet. 
    Your task is to identify and list every food and drink item you can find.
    Provide the output as a simple, clear, bulleted list. Do not add any commentary.
    
    Example:
    - 1 cup black coffee
    - 2 fried eggs
    - 3 strips of bacon
    - 1 apple
    - 1 ham and cheese sandwich
    - 1 can of soda
    """
    
    message_content = [
        {"type": "text", "text": prompt_text}
    ]
    
    # Process each file and add its content to the message
    for file_path in input_files:
        file_content = process_file(file_path)
        if file_content:
            message_content.append(file_content)

    # Create the HumanMessage
    message = HumanMessage(content=message_content)
    
    # Call the vision model
    response = vision_llm.invoke([message])
    
    print(f"Parsed Diet:\n{response.content}")
    return {"parsed_diet": response.content}


def analyze_diet(state: GraphState):
    """
    Node 2: Analyze the parsed food list for habits and nutritional gaps.
    """
    print("--- 2. ANALYZING DIET ---")
    parsed_diet = state["parsed_diet"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful dietary analyst. Your goal is to find patterns."),
        ("human", """
        Based on the following list of consumed foods, please provide a brief, 
        concise analysis of the user's dietary habits. 
        
        Focus on:
        - Macronutrient balance (e.g., high-carb, low-protein)
        - Food groups (e.g., lacking vegetables, high in processed foods)
        - Potential nutritional gaps (e.g., "seems low in fiber", "high sugar intake")
        
        Keep your analysis to 2-3 short paragraphs.
        
        Food List:
        {diet_list}
        """),
    ])
    
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"diet_list": parsed_diet})
    
    print(f"Dietary Analysis:\n{analysis}")
    return {"analysis": analysis}


def retrieve_facts(state: GraphState):
    """
    Node 3: Retrieve relevant nutritional facts from the vector store.
    This is the RAG step to ensure accuracy.
    """
    print("--- 3. RETRIEVING FACTS ---")
    analysis = state["analysis"]
    
    # The analysis becomes the query for our "fact-checking" system
    retrieved_docs = retriever.invoke(analysis)
    
    # Format the retrieved documents into a single string
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"Retrieved Context:\n{context[:500]}...")
    return {"context": context}


def generate_recommendation(state: GraphState):
    """
    Node 4: Generate a final recommendation grounded in the retrieved facts.
    """
    print("--- 4. GENERATING RECOMMENDATION ---")
    analysis = state["analysis"]
    context = state["context"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a supportive and knowledgeable health assistant. 
        Your task is to provide a helpful, non-judgmental food recommendation 
        to a user based on their recent diet.
        
        **IMPORTANT**: You MUST base your recommendations *only* on the 
        "Factual Nutritional Information" provided. Do not invent facts.
        
        Follow this structure:
        1.  Briefly and kindly acknowledge their recent habits (from the "Analysis").
        2.  Provide 1-2 simple, actionable suggestions for their *next meal* or *next day*
            that would help balance their diet.
        3.  Explain *why* this is a good choice, citing the "Factual Nutritional Information".
        """),
        ("human", """
        Here is my situation:
        
        **My Recent Diet Analysis:**
        {analysis}
        
        **Factual Nutritional Information (Your source of truth):**
        {context}
        
        Please provide your recommendation.
        """),
    ])
    
    chain = prompt | llm | StrOutputParser()
    recommendation = chain.invoke({"analysis": analysis, "context": context})
    
    print(f"Final Recommendation:\n{recommendation}")
    return {"recommendation": recommendation}

# --- 4. Define and Compile the Graph ---

print("Assembling the graph...")

workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("analyze_diet", analyze_diet)
workflow.add_node("retrieve_facts", retrieve_facts)
workflow.add_node("generate_recommendation", generate_recommendation)

# Add the edges
workflow.set_entry_point("parse_input")
workflow.add_edge("parse_input", "analyze_diet")
workflow.add_edge("analyze_diet", "retrieve_facts")
workflow.add_edge("retrieve_facts", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)

# Compile the graph
app = workflow.compile()

# --- 5. Run the Application (for command-line testing) ---

if __name__ == "__main__":
    # To run, you need some sample inputs.
    # Create a folder 'sample_inputs'
    # Add 'diet.txt' or 'food1.jpg', 'food2.jpg' etc.
    
    # Create a 'sample_inputs' folder if it doesn't exist
    if not os.path.exists("sample_inputs"):
        os.makedirs("sample_inputs")
    
    # Create a sample 'diet.txt' if it doesn't exist
    sample_diet_path = "sample_inputs/diet.txt"
    if not os.path.exists(sample_diet_path):
        with open(sample_diet_path, "w") as f:
            f.write("""
Day 1:
Breakfast: 2 donuts, 1 large coffee with cream and sugar
Lunch: Pepperoni pizza (3 slices), 1 can of soda
Dinner: 1 steak, 1 baked potato with butter

Day 2:
Breakfast: Skipped
Lunch: 1 cheeseburger, large fries, 1 diet soda
Dinner: Pasta with alfredo sauce, 1 glass of wine
            """)

    # Define the inputs
    inputs = {"input_files": [sample_diet_path]}
    # Or for images:
    # inputs = {"input_files": ["sample_inputs/food1.jpg", "sample_inputs/food2.jpg"]}

    print("🚀 Running the Diet Analysis Agent (Test Run)...")

    # Run the graph
    final_state = app.invoke(inputs)

    print("\n--- ✅ FINAL RECOMMENDATION (Test Run) ---")
    print(final_state["recommendation"])