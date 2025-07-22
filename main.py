import streamlit as st
import os
import json
from datetime import datetime
from multi_agent_health_system import MultiAgentHealthSystem
from extended_schemas import UserInput
from config import PERSONALITIES, DEFAULT_OPENAI_SETTINGS, MEDICAL_DISCLAIMER

st.set_page_config(
    page_title="HealthBot - AI Patient Education",
    page_icon="🏥",
    layout="wide"
)

def save_conversation():
    """Save conversation to JSON file"""
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"health_conversation_{timestamp}.json"
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "settings": {
                "personality": st.session_state.get("personality", "friendly"),
                "openai_settings": st.session_state.get("openai_settings", {})
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        return filename
    return None

def summarize_conversation():
    """Generate conversation summary"""
    if not st.session_state.messages:
        return "No conversation to summarize."
    
    user_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    assistant_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]
    
    summary = f"""
**Conversation Summary**
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Topics Discussed**: {len(user_messages)} health questions
- **Key Symptoms/Questions**: {'; '.join(user_messages[:3])}...
- **Total Messages**: {len(st.session_state.messages)}
"""
    return summary

def load_conversation():
    """Load conversation from uploaded file"""
    uploaded_file = st.file_uploader("Upload conversation file", type="json")
    if uploaded_file:
        try:
            conversation_data = json.load(uploaded_file)
            st.session_state.messages = conversation_data.get("messages", [])
            if "settings" in conversation_data:
                settings = conversation_data["settings"]
                st.session_state.personality = settings.get("personality", "friendly")
                st.session_state.openai_settings = settings.get("openai_settings", {})
            st.success("Conversation loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading conversation: {str(e)}")

def main():
    st.title("🏥 HealthBot - Multi-Agent AI Health System")
    st.markdown("*Advanced multi-agent health education and symptom analysis*")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Personality
        personality = st.selectbox(
            "Agent Personality:",
            options=list(PERSONALITIES.keys()),
            format_func=str.title,
            key="personality_select"
        )
        
        # OpenAI parameters
        st.subheader("AI Parameters")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.1)
        max_tokens = st.slider("Max tokens", 100, 2000, 1000, 100)
        
        openai_settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        st.session_state.openai_settings = openai_settings
        st.session_state.personality = personality
        
        # Multi-Agent Status
        st.subheader("🤖 Multi-Agent System")
        st.info("""
        **Active Agents:**
        - 🚨 Triage Agent
        - 🔍 Diagnosis Agent  
        - 🥗 Diet Agent
        - 💊 Treatment Agent
        - 🧠 Synthesis Agent
        - 🔍 Validation Agent
        """)
        
        # Conversation management
        st.subheader("💾 Conversation")
        
        if st.button("💾 Save"):
            filename = save_conversation()
            if filename:
                st.success(f"Saved: {filename}")
            else:
                st.warning("No conversation to save")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize multi-agent system
    if st.session_state.agent is None:
        st.session_state.agent = MultiAgentHealthSystem(openai_settings)
    else:
        st.session_state.agent.openai_settings = openai_settings
    
    # Medical disclaimer
    with st.expander("⚠️ Important Medical Disclaimer", expanded=False):
        st.error(MEDICAL_DISCLAIMER)
    
    # Main chat interface
    st.subheader("💬 Multi-Agent Health Consultation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show agent details for assistant messages
            if message["role"] == "assistant" and "agent_details" in message:
                with st.expander("🤖 Agent Processing Details"):
                    details = message["agent_details"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Triage Result:**")
                        st.json(details.get("triage_result", {}))
                        
                        st.write("**Diet Recommendations:**")
                        st.json(details.get("diet_result", {}))
                    
                    with col2:
                        st.write("**Diagnosis Analysis:**")
                        st.json(details.get("diagnosis_result", {}))
                        
                        st.write("**Treatment Plan:**")
                        st.json(details.get("treatment_result", {}))
                    
                    st.write("**Validation Status:**", details.get("validation_status", "Unknown"))
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms or health questions..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with multi-agent system
        with st.chat_message("assistant"):
            with st.spinner("Processing through multi-agent system..."):
                try:
                    # Create user input
                    user_input = UserInput(
                        symptoms=prompt,
                        age=None,
                        gender=None
                    )
                    
                    # Process with multi-agent system
                    result = st.session_state.agent.process_health_query(
                        user_input, 
                        personality,
                        thread_id="chat_session"
                    )
                    
                    response = result["response"]
                    
                    # Display response
                    st.write(response)
                    
                    # Add assistant message with agent details
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "agent_details": {
                            "triage_result": result["triage_result"].dict(),
                            "diagnosis_result": result["diagnosis_result"].dict(),
                            "diet_result": result["diet_result"].dict(),
                            "treatment_result": result["treatment_result"].dict(),
                            "validation_status": result["validation_status"]
                        }
                    })
                    
                    # Show processing pipeline
                    with st.expander("🔄 Agent Processing Pipeline"):
                        st.write("✅ Triage Agent - Classification complete")
                        st.write("✅ Diagnosis Agent - Research and analysis complete")
                        st.write("✅ Diet Agent - Nutritional recommendations complete")
                        st.write("✅ Treatment Agent - Care guidance complete")
                        st.write("✅ Synthesis Agent - Plan integration complete")
                        st.write(f"✅ Validation Agent - Status: {result['validation_status']}")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Usage instructions
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to the Multi-Agent HealthBot!** 
        
        This advanced system uses 6 specialized AI agents:
        - **Triage Agent**: Classifies and prioritizes your health concerns
        - **Diagnosis Agent**: Researches symptoms and provides educational insights
        - **Diet Agent**: Offers nutritional recommendations
        - **Treatment Agent**: Suggests care options and next steps
        - **Synthesis Agent**: Combines all recommendations into a comprehensive plan
        - **Validation Agent**: Ensures accuracy and safety
        
        Ask me about symptoms, nutrition, supplements, or general health questions.
        """)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in environment variables")
        st.stop()
    
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("⚠️ TAVILY_API_KEY not found - web search will be limited")
    
    main()