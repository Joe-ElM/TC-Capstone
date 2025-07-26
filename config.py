# import os
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# # API Keys
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# # Personality settings - DEFAULT TO CONCISE
# PERSONALITIES = {
#     "concise": {
#         "description": "Direct and to-the-point responses",
#         "temperature": 0.2,
#         "max_tokens": 500
#     },
#     "friendly": {
#         "description": "Warm and empathetic responses", 
#         "temperature": 0.2,
#         "max_tokens": 800
#     },
#     "professional": {
#         "description": "Clinical and formal responses",
#         "temperature": 0.2,
#         "max_tokens": 600
#     }
# }

# # Default OpenAI settings
# DEFAULT_OPENAI_SETTINGS = {
#     "temperature": 0.2,
#     "top_p": 0.9,
#     "frequency_penalty": 0.0,
#     "max_tokens": 1000
# }

# # Health system prompt
# HEALTH_PROMPT = """You are YOU AI Health Team, an AI health assistant providing educational information.
# Your responses should be:
# - Accurate and evidence-based
# - Personalized to the patient's specific situation
# - Clear and actionable
# - Always emphasizing the importance of professional medical care

# Never diagnose conditions or prescribe medications. Always encourage users to consult healthcare professionals."""

# # Medical disclaimer
# MEDICAL_DISCLAIMER = """
# ⚠️ **EDUCATIONAL PURPOSES ONLY - NOT MEDICAL ADVICE**
# This information is for educational purposes only. Always consult with a qualified healthcare professional for medical concerns. In case of emergency, contact emergency services immediately.
# """

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Enhanced Personality Settings
PERSONALITIES = {
    "concise": {
        "description": "Direct and to-the-point responses",
        "temperature": 0.3,
        "max_tokens": 600,
        "system_prompt": """You are YOU AI Health Team. Provide direct, actionable health guidance.
        - Use bullet points for clarity
        - Focus on essential information only
        - Give specific next steps with timelines
        - Skip lengthy explanations"""
    },
    "friendly": {
        "description": "Warm and empathetic responses", 
        "temperature": 0.4,
        "max_tokens": 1000,
        "system_prompt": """You are YOU AI Health Team, a caring health companion.
        - Use the patient's name when known
        - Acknowledge their concerns with empathy
        - Provide encouragement and hope
        - Explain things in accessible, non-technical language
        - Reference their health journey and progress"""
    },
    "professional": {
        "description": "Clinical and formal responses",
        "temperature": 0.3,
        "max_tokens": 800,
        "system_prompt": """You are YOU AI Health Team, providing clinical guidance.
        - Use medical terminology appropriately
        - Structure responses systematically
        - Include evidence-based recommendations
        - Provide differential considerations
        - Maintain professional tone throughout"""
    }
}


# Default OpenAI settings
DEFAULT_OPENAI_SETTINGS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.1,
    "max_tokens": 1000
}

# Main health system prompt
HEALTH_PROMPT = """You are YOU AI Health Team, an AI health assistant providing personalized educational information.

Core principles:
- Use ALL available patient data (age, conditions, lab values, lifestyle) in responses
- Reference specific numbers (e.g., "Your HbA1c of 6.1%...")
- Build on previous conversations ("As we discussed last time...")
- Provide 2-3 specific, actionable recommendations
- Connect advice to patient's unique situation (menopause, stress patterns, etc.)
- Always emphasize the importance of professional medical care

Never diagnose conditions or prescribe medications."""

# Medical disclaimer
MEDICAL_DISCLAIMER = """
⚠️ **EDUCATIONAL PURPOSES ONLY - NOT MEDICAL ADVICE**
This information is for educational purposes only. Always consult with a qualified healthcare professional for medical concerns. In case of emergency, contact emergency services immediately.
"""