from typing import List, Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

#=============================================================================
# USER INPUT MODELS
#=============================================================================

class UserInput(BaseModel):
    symptoms: str = Field(description="User's described symptoms")
    age: Optional[int] = Field(description="User's age")
    gender: Optional[str] = Field(description="User's gender")
    medical_history: Optional[List[str]] = Field(default=[], description="Previous conditions")
    current_medications: Optional[List[str]] = Field(default=[], description="Current medications")
    allergies: Optional[List[str]] = Field(default=[], description="Known allergies")

#=============================================================================
# AGENT STATE MODELS
#=============================================================================

class TriageResult(BaseModel):
    intent_classification: str
    urgency_level: str  # LOW, MEDIUM, HIGH, EMERGENCY
    emergency_flags: List[str]
    routing_decision: str
    confidence_score: float

class DiagnosisResult(BaseModel):
    symptoms: List[str]
    symptom_analysis: Dict[str, any]
    possible_conditions: List[Dict[str, any]]
    severity_level: str
    recommended_tests: List[str]
    red_flags: List[str]
    research_sources: List[str]

class DietResult(BaseModel):
    diagnosed_condition: str
    dietary_restrictions: List[str]
    nutritional_needs: Dict[str, any]
    recommended_foods: List[str]
    foods_to_avoid: List[str]
    meal_suggestions: List[str]
    supplements: List[str]

class TreatmentResult(BaseModel):
    diagnosis: Dict[str, any]
    treatment_options: List[Dict[str, any]]
    care_recommendations: List[str]
    lifestyle_changes: List[str]
    follow_up_schedule: Dict[str, any]
    when_to_see_doctor: str
    appointment_needed: bool

class SynthesisResult(BaseModel):
    all_agent_outputs: Dict[str, any]
    safety_validations: List[str]
    cross_checks: Dict[str, any]
    final_recommendations: Dict[str, any]
    appointment_needed: bool
    priority_level: str

class HallucinationCheck(BaseModel):
    source_citations: List[str]
    fact_verification: Dict[str, any]
    consistency_score: float
    medical_accuracy: Dict[str, any]
    flagged_claims: List[str]
    validation_status: str  # APPROVED, FLAGGED, REJECTED

#=============================================================================
# MAIN AGENT STATE
#=============================================================================

class MultiAgentHealthState(TypedDict):
    user_input: UserInput
    user_id: Optional[str]
    conversation_id: str
    
    # Agent Results
    triage_result: Optional[TriageResult]
    diagnosis_result: Optional[DiagnosisResult]
    diet_result: Optional[DietResult]
    treatment_result: Optional[TreatmentResult]
    synthesis_result: Optional[SynthesisResult]
    hallucination_check: Optional[HallucinationCheck]
    
    # Shared Context
    research_content: str
    conversation_history: List[str]
    personality: str
    openai_settings: Dict[str, float]
    
    # Final Output
    response: str
    execution_metadata: Dict[str, any]

#=============================================================================
# USER PROFILE MODEL
#=============================================================================

class UserProfile(BaseModel):
    user_id: str
    personal_info: Dict[str, any]  # age, gender, etc.
    medical_history: List[str]
    conversation_summaries: List[Dict[str, any]]
    chronic_conditions: List[str]
    medications: List[str]
    dietary_restrictions: List[str]
    created_at: datetime
    last_updated: datetime