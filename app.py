import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core import Settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import chromadb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== Configuration ==========
VISITOR_DATA_FILE = "visitors_details.csv"
INTENT_DATA_FILE = "user_intents_log.csv"
GROQ_API_KEY = "Your_api_key"  # Your Groq API key
CSV_FILE_PATH = "cleaned_output.csv"  # Your CSV file path

# ========== Session State Setup ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
if "visitor_info" not in st.session_state:
    st.session_state.visitor_info = {
        "name": "",
        "email": "",
        "purpose": "",
        "intents": [],
        "intent_details": [],  # Store detailed intent information
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": str(datetime.now().timestamp()),
        "profile_completed": False
    }

# ========== Enhanced Data Saving Functions ==========
def save_visitor_data():
    """Save visitor data into visitors_details.csv"""
    visitor = st.session_state.visitor_info
    if visitor["name"] and visitor["email"] and visitor["purpose"]:
        intents = visitor.get("intents", [])
        intent_counts = pd.Series(intents).value_counts().to_dict()
        most_searched_intent = pd.Series(intents).mode().iloc[0] if intents else ""
        
        row = {
            "name": visitor["name"],
            "email": visitor["email"],
            "purpose": visitor["purpose"],
            "intent_counts": str(intent_counts),
            "most_searched_intent": most_searched_intent,
            "total_queries": len(intents),
            "timestamp": visitor["timestamp"],
            "session_id": visitor["session_id"]
        }

        file_path = VISITOR_DATA_FILE
        if os.path.exists(file_path):
            df_existing = pd.read_csv(file_path)
            df_combined = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
        else:
            df_combined = pd.DataFrame([row])

        df_combined.to_csv(file_path, index=False)
        return True
    return False

def save_intent_data(query, detected_intent, confidence_score):
    """Save individual intent data into user_intents_log.csv"""
    visitor = st.session_state.visitor_info
    
    intent_row = {
        "session_id": visitor["session_id"],
        "user_name": visitor["name"],
        "user_email": visitor["email"],
        "query": query,
        "detected_intent": detected_intent,
        "confidence_score": confidence_score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    file_path = INTENT_DATA_FILE
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, pd.DataFrame([intent_row])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([intent_row])
    
    df_combined.to_csv(file_path, index=False)

# ========== LlamaIndex Setup Functions ==========
def load_csv_as_documents(csv_file):
    try:
        df = pd.read_csv(csv_file)
        documents = []
        for idx, row in df.iterrows():
            # Create more structured document text
            text = f"Query: {row.get('text', '')} | Intent: {row.get('intent', 'unknown')} | Response: {row.get('response', '')}"
            doc = Document(
                text=text,
                metadata={
                    "intent": row.get('intent', 'unknown'),
                    "response": row.get('response', ''),
                    "original_text": row.get('text', '')
                }
            )
            documents.append(doc)
        return documents
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def setup_llamaindex_with_chroma(persist_dir="chroma_db"):
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_or_create_collection("csv_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Configure settings
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        return vector_store
    except Exception as e:
        st.error(f"Error setting up ChromaDB: {e}")
        return None

def retrieve_context_with_intent(index, query, top_k=3):
    try:
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        if nodes:
            best_node = nodes[0]
            context = best_node.get_content()
            intent = best_node.metadata.get('intent', 'unknown')
            confidence = best_node.score if hasattr(best_node, 'score') else 0.8
            
            return context, intent, confidence
        return "", "unknown", 0.0
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "", "unknown", 0.0

# ========== Enhanced LangChain with Groq ==========
def generate_answer_groq(context, query, intent):
    try:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

        prompt = ChatPromptTemplate.from_template(
            """
            You are an intelligent assistant for Tekworks company, built by the data science team. 
            You help visitors get information about the company based on their intent and queries.
            
            Detected Intent: {intent}
            Context from knowledge base: {context}
            User Query: {question}
            
            Instructions:
            1. Provide a helpful and accurate response based on the context
            2. If the intent is clear, tailor your response to address that specific intent
            3. Be friendly and professional in your tone
            4. If you don't have enough information, politely say so
            5. Keep responses concise but informative
            
            Response:
            """
        )

        chain = prompt | llm
        response = chain.invoke({
            "context": context, 
            "question": query,
            "intent": intent
        })
        return response.content
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

# ========== Initialize System ==========
def initialize_system(csv_file):
    if st.session_state.index is None:
        with st.spinner("🔄 Initializing AI system..."):
            # Load documents
            docs = load_csv_as_documents(csv_file)
            if docs is None:
                return False
            
            # Setup Chroma
            vector_store = setup_llamaindex_with_chroma()
            if vector_store is None:
                return False
            
            # Create index
            try:
                index = VectorStoreIndex.from_documents(
                    documents=docs,
                    vector_store=vector_store
                )
                st.session_state.vector_store = vector_store
                st.session_state.index = index
                return True
            except Exception as e:
                st.error(f"Error creating index: {e}")
                return False
    return True

# ========== Enhanced Visualization Functions ==========
def create_intent_visualization():
    """Create enhanced intent visualization using Plotly"""
    intents = st.session_state.visitor_info["intents"]
    intent_details = st.session_state.visitor_info["intent_details"]
    
    if not intents:
        return None, None
    
    # Intent frequency chart
    intent_counts = pd.Series(intents).value_counts()
    
    fig_bar = px.bar(
        x=intent_counts.index,
        y=intent_counts.values,
        title="🎯 Intent Distribution",
        labels={'x': 'Intent Categories', 'y': 'Frequency'},
        color=intent_counts.values,
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        xaxis_tickangle=-45
    )
    
    # Intent confidence over time
    if intent_details:
        df_details = pd.DataFrame(intent_details)
        fig_line = px.line(
            df_details,
            x=range(len(df_details)),
            y='confidence',
            color='intent',
            title="🎯 Intent Confidence Over Time",
            labels={'x': 'Query Number', 'y': 'Confidence Score'}
        )
        fig_line.update_layout(height=300, title_font_size=16)
        return fig_bar, fig_line
    
    return fig_bar, None

def create_session_summary():
    """Create session summary statistics"""
    intents = st.session_state.visitor_info["intents"]
    intent_details = st.session_state.visitor_info["intent_details"]
    
    if not intents:
        return {}
    
    summary = {
        "Total Queries": len(intents),
        "Unique Intents": len(set(intents)),
        "Most Common Intent": pd.Series(intents).mode().iloc[0] if intents else "None",
        "Average Confidence": np.mean([d['confidence'] for d in intent_details]) if intent_details else 0
    }
    
    return summary

# ========== Streamlit UI ==========
st.set_page_config(
    page_title="🧠 Tekworks AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Tekworks Company Information Chatbot")
st.markdown("### 🤖 AI-Powered Intent Detection & Company Information Assistant")

# Initialize the system
if not initialize_system(CSV_FILE_PATH):
    st.error("❌ System initialization failed. Please check the logs.")
    st.stop()

# Profile completion form with enhanced UI
if not st.session_state.visitor_info["profile_completed"]:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### 👤 Welcome! Please Complete Your Profile")
            st.markdown("*Required to access the AI assistant*")
            
            with st.form("profile_form", clear_on_submit=True):
                name = st.text_input("👤 Your Full Name*", placeholder="Enter your full name")
                email = st.text_input("📧 Email Address*", placeholder="your.email@company.com")
                purpose = st.text_area(
                    "🎯 Purpose of Visit*",
                    placeholder="Describe why you're here (e.g., job inquiry, partnership, general information)",
                    height=100
                )
                
                submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
                with submit_col2:
                    if st.form_submit_button("🚀 Start Chatting", use_container_width=True):
                        if name and email and purpose:
                            st.session_state.visitor_info.update({
                                "name": name,
                                "email": email,
                                "purpose": purpose,
                                "profile_completed": True
                            })
                            st.success("✅ Profile completed! You can now start chatting.")
                            st.rerun()
                        else:
                            st.error("⚠️ Please fill all required fields (marked with *)")

# Main chat interface (only shown if profile is completed)
if st.session_state.visitor_info["profile_completed"]:
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        st.info(f"""
        **Name:** {st.session_state.visitor_info['name']}  
        **Email:** {st.session_state.visitor_info['email']}  
        **Purpose:** {st.session_state.visitor_info['purpose']}
        """)
        
        # Save buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session", use_container_width=True):
                if save_visitor_data():
                    st.success("✅ Session saved!")
                else:
                    st.error("❌ Save failed!")
        
        with col2:
            if st.button("📊 Export Data", use_container_width=True):
                # Create downloadable CSV
                if st.session_state.visitor_info["intent_details"]:
                    df_export = pd.DataFrame(st.session_state.visitor_info["intent_details"])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"intents_{st.session_state.visitor_info['session_id']}.csv",
                        mime="text/csv"
                    )
        
        st.markdown("---")
        
        # Session Summary with improved styling
        summary = create_session_summary()
        if summary:
            st.markdown("### 📊 Session Summary")
            # Create a container with custom styling
            with st.container():
                st.markdown("""
                <style>
                    .summary-container {
                        background-color: #2a2a2a;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                    }
                    .summary-metric {
                        color: white;
                        margin-bottom: 10px;
                    }
                    .summary-metric h4 {
                        color: #a1a1a1;
                        margin-bottom: 5px;
                        font-size: 14px;
                    }
                    .summary-metric p {
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        margin-top: 0;
                    }
                </style>
                <div class="summary-container">
                """, unsafe_allow_html=True)
                
                for key, value in summary.items():
                    # Handle float formatting separately
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else value
                    st.markdown(f"""
                    <div class="summary-metric">
                        <h4>{key}</h4>
                        <p>{formatted_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Intent Visualization
        if st.session_state.visitor_info["intents"]:
            st.markdown("### 🎯 Intent Analytics")
            
            # Create visualizations
            fig_bar, fig_line = create_intent_visualization()
            
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
            
            if fig_line:
                st.plotly_chart(fig_line, use_container_width=True)

    # Main chat area with enhanced layout
    st.markdown("---")
    
    # Chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "intent" in message and message["role"] == "assistant":
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.caption(f"🎯 **Intent:** {message['intent']}")
                    with col2:
                        confidence = message.get('confidence', 0)
                        color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.caption(f"📊 **Confidence:** :{color}[{confidence:.2f}]")
                    with col3:
                        st.caption(f"⏱️ Query #{i//2 + 1}")

    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about Tekworks..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.spinner("🔍 Analyzing your query and searching knowledge base..."):
            try:
                # Retrieve context and detect intent
                context, detected_intent, confidence = retrieve_context_with_intent(
                    st.session_state.index, prompt
                )
                
                # Save intent data
                save_intent_data(prompt, detected_intent, confidence)
                
                # Track intent in session
                st.session_state.visitor_info["intents"].append(detected_intent)
                st.session_state.visitor_info["intent_details"].append({
                    "query": prompt,
                    "intent": detected_intent,
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Generate AI response
                answer = generate_answer_groq(context, prompt, detected_intent)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Enhanced response metadata
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.caption(f"🎯 **Intent:** {detected_intent}")
                    with col2:
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.caption(f"📊 **Confidence:** :{confidence_color}[{confidence:.2f}]")
                    with col3:
                        query_num = len([m for m in st.session_state.messages if m["role"] == "user"])
                        st.caption(f"⏱️ Query #{query_num}")
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "intent": detected_intent,
                    "confidence": confidence
                })
                
            except Exception as e:
                st.error(f"❌ Error processing your request: {e}")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 <strong>Tekworks AI Assistant</strong> | Built with ❤️ by the Data Science Team</p>
    <p><em>Powered by LlamaIndex 🦙 + LangChain 🔗 + Groq ⚡</em></p>
</div>
""", unsafe_allow_html=True)

# Enhanced Custom CSS
st.markdown("""
    <style>
        /* Main content styles */
        .main .block-container {
            padding-top: 2rem;
            max-width: 95%;
        }
        
        /* Chat message styles */
        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stChatMessage[data-testid="user"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .stChatMessage[data-testid="assistant"] {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        /* Sidebar styles */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%);
            color: white;
        }
        
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li {
            color: white;
        }
        
        [data-testid="stSidebar"] .stButton button {
            background-color: #444;
            color: white;
            border: 1px solid #666;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #555;
            border-color: #777;
        }
        
        [data-testid="stSidebar"] .stInfo {
            background-color: rgba(255,255,255,0.1);
            border-left: 4px solid #667eea;
        }
        
        /* Metric styles */
        .stMetric {
            background: rgba(255,255,255,0.1);
            padding: 0.75rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }
        
        .stMetric label {
            color: #aaa !important;
        }
        
        .stMetric div {
            color: white !important;
        }
        
        /* General button styles */
        .stButton button {
            border-radius: 20px;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        /* Form input styles */
        [data-testid="stTextInput"] label:after, 
        [data-testid="stTextArea"] label:after {
            content: " *";
            color: red;
        }
        
        /* Plot container styles */
        .plot-container {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)