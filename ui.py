 import streamlit as st
import os
import tempfile
from PIL import Image

# Import the compiled graph 'app' from your app.py file
try:
    from app import app
except ImportError:
    st.error("Error: Could not import 'app' from app.py. Please ensure app.py is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading app.py. Make sure all dependencies are installed and .env is set up. Error: {e}")
    st.stop()


# --- Helper Function ---

def save_uploaded_file(uploaded_file):
    """Saves an uploaded file to a temporary file and returns the path."""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def save_text_input(text_content):
    """Saves text area content to a temporary file and returns the path."""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "typed_diet.txt")
        
        # Write the text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return file_path
    except Exception as e:
        st.error(f"Error saving text input: {e}")
        return None

def cleanup_temp_files(file_paths):
    """Removes temporary files and their directories."""
    for file_path in file_paths:
        try:
            dir_name = os.path.dirname(file_path)
            os.remove(file_path)
            os.rmdir(dir_name)
        except Exception as e:
            print(f"Warning: Could not clean up temp file {file_path}. Error: {e}")

# --- UI Layout ---

st.set_page_config(page_title="🍏 AI Diet Analyzer", layout="wide")
st.title("🍏 Your Personal AI-Powered Diet Analyzer")
st.markdown("Provide your recent diet information below, and I'll analyze it and provide some healthy recommendations.")

tab1, tab2 = st.tabs(["➡️ Text Input (Type or Upload .txt)", "📸 Image Upload"])

# --- Tab 1: Text Input ---
with tab1:
    st.header("Provide Your Diet as Text")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Type Your Diet")
        text_input = st.text_area("List all foods and drinks you've had recently:", height=300, 
                                  placeholder="Example:\nDay 1:\n- Coffee with cream\n- 2 eggs, 3 strips of bacon\n...\nDay 2:\n- Apple\n- Ham and cheese sandwich\n...")
    
    with col2:
        st.subheader("...or Upload a Text File")
        text_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if st.button("Analyze Text Input", key="text_button"):
        file_paths = []
        if text_input.strip():
            # User typed in the text area
            st.info("Using text from the text area...")
            temp_path = save_text_input(text_input)
            if temp_path:
                file_paths.append(temp_path)
        
        elif text_file is not None:
            # User uploaded a file
            st.info(f"Using uploaded file: {text_file.name}")
            temp_path = save_uploaded_file(text_file)
            if temp_path:
                file_paths.append(temp_path)
        
        else:
            st.warning("Please either type your diet or upload a .txt file.")
        
        if file_paths:
            with st.spinner("🧠 Analyzing your diet... Please wait."):
                try:
                    # Run the graph
                    inputs = {"input_files": file_paths}
                    final_state = app.invoke(inputs)
                    
                    # Display results
                    st.subheader("✅ Analysis Complete!")
                    st.divider()
                    st.subheader("Your Dietary Analysis:")
                    st.write(final_state.get("analysis", "No analysis generated."))
                    st.divider()
                    st.subheader("My Recommendations for You:")
                    st.write(final_state.get("recommendation", "No recommendation generated."))

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                finally:
                    # Clean up the temporary files
                    cleanup_temp_files(file_paths)

# --- Tab 2: Image Upload ---
with tab2:
    st.header("Upload Images of Your Meals")
    image_files = st.file_uploader("You can upload multiple images (.jpg, .png)", 
                                   type=["jpg", "jpeg", "png"], 
                                   accept_multiple_files=True)
    
    if image_files:
        st.write(f"You have uploaded **{len(image_files)}** images.")
        
        # Display image previews
        image_cols = st.columns(min(len(image_files), 5))
        for idx, img_file in enumerate(image_files):
            with image_cols[idx % 5]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_column_width=True)
    
    if st.button("Analyze Images", key="image_button"):
        if image_files:
            file_paths = []
            for img_file in image_files:
                temp_path = save_uploaded_file(img_file)
                if temp_path:
                    file_paths.append(temp_path)
            
            if file_paths:
                with st.spinner("📸 Analyzing images... This may take a moment."):
                    try:
                        # Run the graph
                        inputs = {"input_files": file_paths}
                        final_state = app.invoke(inputs)
                        
                        # Display results
                        st.subheader("✅ Analysis Complete!")
                        st.divider()
                        st.subheader("Here's what I identified:")
                        st.write(final_state.get("parsed_diet", "Could not parse diet from images."))
                        st.divider()
                        st.subheader("Your Dietary Analysis:")
                        st.write(final_state.get("analysis", "No analysis generated."))
                        st.divider()
                        st.subheader("My Recommendations for You:")
                        st.write(final_state.get("recommendation", "No recommendation generated."))

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                    finally:
                        # Clean up the temporary files
                        cleanup_temp_files(file_paths)
        else:
            st.warning("Please upload at least one image.")