import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config import TAVILY_API_KEY

#=============================================================================
# TOOL SCHEMAS
#=============================================================================

class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for retrieval")

class BMIInput(BaseModel):
    height_cm: float = Field(description="Height in centimeters")
    weight_kg: float = Field(description="Weight in kilograms")

class NutritionInput(BaseModel):
    condition: str = Field(description="Medical condition")
    age: int = Field(description="Patient age")
    gender: str = Field(description="Patient gender")
    activity_level: str = Field(description="Activity level")

class AppointmentRequest(BaseModel):
    urgency: str = Field(description="Appointment urgency level")
    condition: str = Field(description="Primary health concern")
    preferred_timeframe: str = Field(description="When patient wants appointment")

#=============================================================================
# RESEARCH TOOLS
#=============================================================================

@tool
def search_wikipedia(query: str, max_docs: int = 2) -> str:
    """Search Wikipedia for medical information"""
    try:
        search_query = f"{query} medical symptoms causes treatment"
        loader = WikipediaLoader(query=search_query, load_max_docs=max_docs)
        docs = loader.load()
        
        formatted_docs = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ])
        
        return formatted_docs
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool
def search_tavily(query: str, max_results: int = 3) -> str:
    """Search web using Tavily for current medical information"""
    try:
        if not TAVILY_API_KEY:
            return "Tavily search unavailable - API key not configured"
            
        tavily_search = TavilySearchResults(
            max_results=max_results,
            api_key=TAVILY_API_KEY
        )
        
        search_query = f"{query} symptoms medical information causes treatment"
        search_docs = tavily_search.invoke(search_query)
        
        formatted_docs = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ])
        
        return formatted_docs
        
    except Exception as e:
        return f"Error searching Tavily: {str(e)}"

#=============================================================================
# CALCULATION TOOLS
#=============================================================================

@tool
def calculate_bmi(height_cm: float, weight_kg: float) -> Dict[str, any]:
    """Calculate BMI and provide health category"""
    try:
        if height_cm <= 0 or weight_kg <= 0:
            return {"error": "Invalid height or weight values"}
        
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
            health_note = "May indicate malnutrition or underlying health issues"
        elif bmi < 25:
            category = "Normal weight"
            health_note = "Healthy weight range"
        elif bmi < 30:
            category = "Overweight"
            health_note = "May increase risk of health problems"
        else:
            category = "Obese"
            health_note = "Associated with increased health risks"
        
        return {
            "bmi": round(bmi, 1),
            "category": category,
            "health_note": health_note,
            "healthy_range": "18.5 - 24.9"
        }
        
    except Exception as e:
        return {"error": f"BMI calculation error: {str(e)}"}

@tool
def calculate_nutrition_needs(condition: str, age: int, gender: str, activity_level: str) -> Dict[str, any]:
    """Calculate basic nutritional needs based on condition and demographics"""
    try:
        # Base caloric needs (simplified calculation)
        if gender.lower() == "male":
            bmr = 88.362 + (13.397 * 70) + (4.799 * 175) - (5.677 * age)  # Using average weight/height
        else:
            bmr = 447.593 + (9.247 * 60) + (3.098 * 165) - (4.330 * age)  # Using average weight/height
        
        activity_multiplier = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very active": 1.9
        }.get(activity_level.lower(), 1.55)
        
        daily_calories = int(bmr * activity_multiplier)
        
        # Condition-specific adjustments
        condition_notes = {}
        if "diabetes" in condition.lower():
            condition_notes["carbohydrates"] = "Focus on complex carbs, limit simple sugars"
            condition_notes["fiber"] = "Aim for 25-35g daily"
        elif "hypertension" in condition.lower():
            condition_notes["sodium"] = "Limit to under 2300mg daily"
            condition_notes["potassium"] = "Increase potassium-rich foods"
        elif "heart" in condition.lower():
            condition_notes["fats"] = "Focus on healthy fats, limit saturated fats"
            condition_notes["omega3"] = "Include omega-3 fatty acids"
        
        return {
            "daily_calories": daily_calories,
            "protein_g": int(daily_calories * 0.15 / 4),  # 15% of calories from protein
            "carbs_g": int(daily_calories * 0.55 / 4),    # 55% from carbs
            "fat_g": int(daily_calories * 0.30 / 9),      # 30% from fats
            "condition_notes": condition_notes,
            "water_liters": round(30 * 70 / 1000, 1)     # 30ml per kg body weight
        }
        
    except Exception as e:
        return {"error": f"Nutrition calculation error: {str(e)}"}

#=============================================================================
# SEVERITY SCORING TOOLS
#=============================================================================

@tool
def score_symptom_severity(symptoms: List[str], age: int = None) -> Dict[str, any]:
    """Score symptom severity based on common medical criteria"""
    try:
        emergency_keywords = [
            "chest pain", "difficulty breathing", "severe bleeding", "unconscious",
            "seizure", "stroke", "heart attack", "severe abdominal pain"
        ]
        
        high_severity = [
            "fever over 101", "persistent vomiting", "severe headache",
            "confusion", "rapid heartbeat", "shortness of breath"
        ]
        
        moderate_severity = [
            "fever", "headache", "nausea", "dizziness", "fatigue",
            "muscle pain", "joint pain"
        ]
        
        symptom_text = " ".join(symptoms).lower()
        score = 0
        severity_factors = []
        
        # Check for emergency symptoms
        for keyword in emergency_keywords:
            if keyword in symptom_text:
                score += 10
                severity_factors.append(f"Emergency: {keyword}")
        
        # Check for high severity symptoms
        for keyword in high_severity:
            if keyword in symptom_text:
                score += 5
                severity_factors.append(f"High: {keyword}")
        
        # Check for moderate symptoms
        for keyword in moderate_severity:
            if keyword in symptom_text:
                score += 2
                severity_factors.append(f"Moderate: {keyword}")
        
        # Age factor
        if age and age > 65:
            score += 2
            severity_factors.append("Age factor: 65+")
        
        # Determine severity level
        if score >= 10:
            level = "EMERGENCY"
            recommendation = "Seek immediate medical attention"
        elif score >= 7:
            level = "HIGH"
            recommendation = "See doctor within 24 hours"
        elif score >= 4:
            level = "MODERATE"
            recommendation = "Schedule appointment within week"
        else:
            level = "LOW"
            recommendation = "Monitor symptoms, self-care measures"
        
        return {
            "severity_score": score,
            "severity_level": level,
            "recommendation": recommendation,
            "factors": severity_factors
        }
        
    except Exception as e:
        return {"error": f"Severity scoring error: {str(e)}"}

#=============================================================================
# APPOINTMENT SCHEDULING TOOLS
#=============================================================================

@tool
def schedule_appointment(urgency: str, condition: str, preferred_timeframe: str) -> Dict[str, any]:
    """Generate appointment scheduling recommendation"""
    try:
        urgency_mapping = {
            "emergency": {"timeframe": "Immediate", "provider": "Emergency Room"},
            "urgent": {"timeframe": "Within 24 hours", "provider": "Urgent Care or Primary Care"},
            "high": {"timeframe": "Within 3-5 days", "provider": "Primary Care"},
            "moderate": {"timeframe": "Within 1-2 weeks", "provider": "Primary Care"},
            "low": {"timeframe": "Within 4 weeks", "provider": "Primary Care"},
            "routine": {"timeframe": "Within 1-3 months", "provider": "Primary Care"}
        }
        
        appointment_info = urgency_mapping.get(urgency.lower(), urgency_mapping["moderate"])
        
        # Condition-specific provider recommendations
        specialist_conditions = {
            "heart": "Cardiologist",
            "skin": "Dermatologist", 
            "diabetes": "Endocrinologist",
            "mental health": "Mental Health Professional",
            "eye": "Ophthalmologist",
            "bone": "Orthopedist"
        }
        
        for key, specialist in specialist_conditions.items():
            if key in condition.lower():
                appointment_info["specialist"] = specialist
                break
        
        return {
            "recommended_timeframe": appointment_info["timeframe"],
            "provider_type": appointment_info["provider"],
            "specialist_referral": appointment_info.get("specialist", "Not required"),
            "scheduling_note": f"For {condition}: {preferred_timeframe}",
            "preparation": [
                "List all current symptoms",
                "Bring medication list",
                "Prepare medical history questions"
            ]
        }
        
    except Exception as e:
        return {"error": f"Appointment scheduling error: {str(e)}"}

#=============================================================================
# SAFETY VALIDATION TOOLS  
#=============================================================================

@tool
def validate_medical_safety(recommendations: Dict[str, any], user_profile: Dict[str, any] = None) -> Dict[str, any]:
    """Validate medical recommendations for safety concerns"""
    try:
        safety_flags = []
        warnings = []
        
        # Check for dangerous self-medication advice
        if "medication" in str(recommendations).lower():
            safety_flags.append("Contains medication recommendations - requires professional oversight")
        
        # Check for emergency symptoms being downplayed
        emergency_terms = ["chest pain", "difficulty breathing", "severe bleeding"]
        rec_text = str(recommendations).lower()
        
        for term in emergency_terms:
            if term in rec_text and "emergency" not in rec_text:
                safety_flags.append(f"Emergency symptom '{term}' may need immediate attention")
        
        # Check age-related concerns
        if user_profile and user_profile.get("age", 0) > 65:
            warnings.append("Elderly patients may have different risk factors")
        
        # Check medication interactions if user profile available
        if user_profile and user_profile.get("medications"):
            warnings.append("Consider current medication interactions")
        
        validation_status = "APPROVED" if not safety_flags else "FLAGGED"
        
        return {
            "validation_status": validation_status,
            "safety_flags": safety_flags,
            "warnings": warnings,
            "requires_disclaimer": True,
            "confidence_level": "High" if not safety_flags else "Medium"
        }
        
    except Exception as e:
        return {"error": f"Safety validation error: {str(e)}"}

#=============================================================================
# TOOL COMBINATION HELPERS
#=============================================================================

def combine_search_results(wikipedia_results: str, tavily_results: str) -> str:
    """Combine and format search results from multiple sources"""
    combined = []
    
    if wikipedia_results and "Error" not in wikipedia_results:
        combined.append("## Wikipedia Sources:\n" + wikipedia_results)
    
    if tavily_results and "Error" not in tavily_results:
        combined.append("## Web Sources:\n" + tavily_results)
    
    if not combined:
        return "No reliable medical information found. Please consult a healthcare professional."
    
    return "\n\n".join(combined)