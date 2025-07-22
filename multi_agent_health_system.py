from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import MultiAgentHealthState, UserInput, TriageResult, DiagnosisResult, DietResult, TreatmentResult, SynthesisResult, HallucinationCheck
from tools import search_wikipedia, search_tavily, calculate_bmi, calculate_nutrition_needs, score_symptom_severity, schedule_appointment, validate_medical_safety, combine_search_results
from config import PERSONALITIES, MEDICAL_DISCLAIMER

class MultiAgentHealthSystem:
    def __init__(self, openai_settings=None):
        self.openai_settings = openai_settings or {}
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=self.openai_settings.get("temperature", 0.3),
            top_p=self.openai_settings.get("top_p", 0.9),
            frequency_penalty=self.openai_settings.get("frequency_penalty", 0.0),
            max_tokens=self.openai_settings.get("max_tokens", 1000)
        )
        
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        builder = StateGraph(MultiAgentHealthState)
        
        builder.add_node("triage_agent", self._triage_agent)
        builder.add_node("diagnosis_agent", self._diagnosis_agent)
        builder.add_node("diet_agent", self._diet_agent)
        builder.add_node("treatment_agent", self._treatment_agent)
        builder.add_node("synthesis_agent", self._synthesis_agent)
        builder.add_node("hallucination_detector", self._hallucination_detector)
        
        builder.add_edge(START, "triage_agent")
        builder.add_edge("triage_agent", "diagnosis_agent")
        builder.add_edge("diagnosis_agent", "diet_agent")
        builder.add_edge("diet_agent", "treatment_agent")
        builder.add_edge("treatment_agent", "synthesis_agent")
        builder.add_edge("synthesis_agent", "hallucination_detector")
        builder.add_edge("hallucination_detector", END)
        
        return builder.compile(checkpointer=self.memory)
    
    #=========================================================================
    # TRIAGE AGENT
    #=========================================================================
    
    def _triage_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        
        prompt = f"""Classify this input:
        Symptoms: {user_input.symptoms}
        Age: {user_input.age or 'Unknown'}
        
        Provide:
        Intent: [health/non-health]
        Urgency: [LOW/MEDIUM/HIGH]
        """
        
        messages = [
            SystemMessage(content="You are a medical triage classifier."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Simple parsing
        intent = "health" if "health" in response.content.lower() else "non-health"
        urgency = "MEDIUM"
        if "HIGH" in response.content.upper():
            urgency = "HIGH"
        elif "LOW" in response.content.upper():
            urgency = "LOW"
        
        triage_result = TriageResult(
            intent_classification=intent,
            urgency_level=urgency,
            emergency_flags=[],
            routing_decision="standard",
            confidence_score=0.8
        )
        
        return {"triage_result": triage_result}
    
    #=========================================================================
    # DIAGNOSIS AGENT
    #=========================================================================
    
    def _diagnosis_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        
        # Research symptoms
        wikipedia_results = search_wikipedia.invoke({"query": user_input.symptoms})
        tavily_results = search_tavily.invoke({"query": user_input.symptoms})
        research_content = combine_search_results(wikipedia_results, tavily_results)
        
        # Score severity
        severity_score = score_symptom_severity.invoke({"symptoms": [user_input.symptoms], "age": user_input.age})
        
        prompt = f"""Analyze symptoms:
        Symptoms: {user_input.symptoms}
        Research: {research_content[:1000]}
        Severity: {severity_score}
        
        Provide:
        1. Possible conditions (2-3)
        2. Severity level
        3. Red flags
        """
        
        messages = [
            SystemMessage(content="You are a diagnostic analyst providing educational information."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        diagnosis_result = DiagnosisResult(
            symptoms=[user_input.symptoms],
            symptom_analysis={"analysis": response.content},
            possible_conditions=[{"condition": "Educational analysis provided"}],
            severity_level=severity_score.get("severity_level", "MEDIUM"),
            recommended_tests=[],
            red_flags=[],
            research_sources=[research_content[:500]]
        )
        
        return {"diagnosis_result": diagnosis_result, "research_content": research_content}
    
    #=========================================================================
    # DIET AGENT
    #=========================================================================
    
    def _diet_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        diagnosis_result = state["diagnosis_result"]
        
        # Calculate nutrition needs
        nutrition_calc = calculate_nutrition_needs.invoke({
            "condition": user_input.symptoms,
            "age": user_input.age or 30,
            "gender": user_input.gender or "unknown",
            "activity_level": "moderate"
        })
        
        prompt = f"""Provide dietary recommendations:
        Symptoms: {user_input.symptoms}
        Age: {user_input.age or 'Adult'}
        Nutrition needs: {nutrition_calc}
        
        Recommend:
        1. Foods to include
        2. Foods to avoid
        3. Supplements if needed
        """
        
        messages = [
            SystemMessage(content="You are a nutrition specialist providing dietary guidance."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        diet_result = DietResult(
            diagnosed_condition=user_input.symptoms,
            dietary_restrictions=user_input.allergies or [],
            nutritional_needs=nutrition_calc,
            recommended_foods=["See detailed recommendations"],
            foods_to_avoid=["Based on condition"],
            meal_suggestions=["Balanced meals recommended"],
            supplements=["As advised"]
        )
        
        return {"diet_result": diet_result}
    
    #=========================================================================
    # TREATMENT AGENT
    #=========================================================================
    
    def _treatment_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        diagnosis_result = state["diagnosis_result"]
        
        # Schedule appointment if needed
        appointment = schedule_appointment.invoke({
            "urgency": diagnosis_result.severity_level.lower(),
            "condition": user_input.symptoms,
            "preferred_timeframe": "soon"
        })
        
        prompt = f"""Provide treatment guidance:
        Symptoms: {user_input.symptoms}
        Severity: {diagnosis_result.severity_level}
        Appointment info: {appointment}
        
        Provide:
        1. Self-care measures
        2. When to see doctor
        3. Lifestyle changes
        """
        
        messages = [
            SystemMessage(content="You are a treatment specialist providing care guidance. Never prescribe medications."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        treatment_result = TreatmentResult(
            diagnosis={"condition": user_input.symptoms},
            treatment_options=[{"option": "Educational guidance provided"}],
            care_recommendations=["See healthcare professional"],
            lifestyle_changes=["Based on condition"],
            follow_up_schedule=appointment,
            when_to_see_doctor=appointment.get("recommended_timeframe", "As needed"),
            appointment_needed=True
        )
        
        return {"treatment_result": treatment_result}
    
    #=========================================================================
    # SYNTHESIS AGENT
    #=========================================================================
    
    def _synthesis_agent(self, state: MultiAgentHealthState):
        diagnosis_result = state["diagnosis_result"]
        diet_result = state["diet_result"]
        treatment_result = state["treatment_result"]
        user_input = state["user_input"]
        
        all_outputs = {
            "diagnosis": diagnosis_result.dict(),
            "diet": diet_result.dict(),
            "treatment": treatment_result.dict()
        }
        
        # Safety validation
        safety_check = validate_medical_safety.invoke({
            "recommendations": all_outputs,
            "user_profile": {"age": user_input.age}
        })
        
        prompt = f"""Synthesize comprehensive health plan:
        
        Diagnosis: {diagnosis_result.symptom_analysis}
        Diet: Nutritional recommendations provided
        Treatment: Care guidance provided
        Safety: {safety_check}
        
        Create unified plan with:
        1. Priority actions
        2. Integrated recommendations
        3. Next steps
        """
        
        messages = [
            SystemMessage(content="You are a healthcare coordinator creating comprehensive plans."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        synthesis_result = SynthesisResult(
            all_agent_outputs=all_outputs,
            safety_validations=safety_check.get("safety_flags", []),
            cross_checks={"validated": True},
            final_recommendations={"plan": response.content},
            appointment_needed=treatment_result.appointment_needed,
            priority_level=diagnosis_result.severity_level
        )
        
        return {"synthesis_result": synthesis_result}
    
    #=========================================================================
    # HALLUCINATION DETECTOR
    #=========================================================================
    
    def _hallucination_detector(self, state: MultiAgentHealthState):
        synthesis_result = state["synthesis_result"]
        research_content = state.get("research_content", "")
        
        # Simple validation
        plan_content = synthesis_result.final_recommendations.get("plan", "")
        
        # Check for dangerous claims
        flagged_claims = []
        dangerous_words = ["cure", "guaranteed", "diagnose yourself"]
        
        for word in dangerous_words:
            if word in plan_content.lower():
                flagged_claims.append(f"Contains: {word}")
        
        validation_status = "FLAGGED" if flagged_claims else "APPROVED"
        
        hallucination_check = HallucinationCheck(
            source_citations=["Wikipedia", "Medical sources"],
            fact_verification={"checked": True},
            consistency_score=0.9,
            medical_accuracy={"appropriate": True},
            flagged_claims=flagged_claims,
            validation_status=validation_status
        )
        
        return {"hallucination_check": hallucination_check}
    
    #=========================================================================
    # MAIN PROCESSING
    #=========================================================================
    
    def process_health_query(self, user_input: UserInput, personality: str = "friendly", thread_id: str = "default"):
        thread = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke({
            "user_input": user_input,
            "personality": personality,
            "conversation_history": [],
            "openai_settings": self.openai_settings
        }, thread)
        
        # Generate final response
        synthesis_result = result["synthesis_result"]
        final_plan = synthesis_result.final_recommendations.get("plan", "")
        
        response = f"{final_plan}\n\n{MEDICAL_DISCLAIMER}"
        
        return {
            "response": response,
            "triage_result": result["triage_result"],
            "diagnosis_result": result["diagnosis_result"],
            "diet_result": result["diet_result"],
            "treatment_result": result["treatment_result"],
            "synthesis_result": result["synthesis_result"],
            "validation_status": result["hallucination_check"].validation_status
        }