from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from extended_schemas import MultiAgentHealthState, UserInput, TriageResult, DiagnosisResult, DietResult, TreatmentResult, SynthesisResult, HallucinationCheck
from extended_tools import search_wikipedia, search_tavily, calculate_bmi, calculate_nutrition_needs, score_symptom_severity, schedule_appointment, validate_medical_safety, combine_search_results
from config import PERSONALITIES, MEDICAL_DISCLAIMER
import re

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
        # Store conversation history at instance level
        self.conversation_history = []
    
    def _build_graph(self):
        builder = StateGraph(MultiAgentHealthState)
        
        # Add all nodes
        builder.add_node("context_resolver", self._context_resolver)
        builder.add_node("triage_agent", self._triage_agent)
        builder.add_node("diagnosis_agent", self._diagnosis_agent)
        builder.add_node("diet_agent", self._diet_agent)
        builder.add_node("treatment_agent", self._treatment_agent)
        builder.add_node("synthesis_agent", self._synthesis_agent)
        builder.add_node("hallucination_detector", self._hallucination_detector)
        
        # Start with context resolver
        builder.add_edge(START, "context_resolver")
        builder.add_edge("context_resolver", "triage_agent")
        
        # Add conditional routing from triage
        builder.add_conditional_edges(
            "triage_agent",
            self._route_decision,
            {
                "diagnosis_only": "diagnosis_agent",
                "diet_only": "diet_agent",
                "treatment_only": "treatment_agent",
                "full_pipeline": "diagnosis_agent",
                "clarification": "synthesis_agent"
            }
        )
        
        # Conditional edges for partial routes
        builder.add_conditional_edges(
            "diagnosis_agent",
            lambda x: "synthesis_agent" if x.get("triage_result", {}).routing_decision == "diagnosis_only" else "diet_agent"
        )
        
        builder.add_conditional_edges(
            "diet_agent",
            lambda x: "synthesis_agent" if x.get("triage_result", {}).routing_decision in ["diet_only", "diagnosis_only"] else "treatment_agent"
        )
        
        builder.add_conditional_edges(
            "treatment_agent",
            lambda x: "synthesis_agent"
        )
        
        # Always validate after synthesis
        builder.add_edge("synthesis_agent", "hallucination_detector")
        builder.add_edge("hallucination_detector", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _route_decision(self, state: MultiAgentHealthState):
        """Determine which route to take based on triage results"""
        triage_result = state.get("triage_result", {})
        return triage_result.routing_decision
    
    #=========================================================================
    # CONTEXT RESOLVER - NEW
    #=========================================================================
    
    def _context_resolver(self, state: MultiAgentHealthState):
        """Resolve pronouns and ambiguous references using conversation history"""
        user_input = state["user_input"]
        conversation_history = state.get("conversation_history", [])
        
        # Check if input contains pronouns or references
        pronouns = ["it", "this", "that", "them", "these", "those"]
        input_lower = user_input.symptoms.lower()
        
        needs_context = any(pronoun in input_lower.split() for pronoun in pronouns)
        
        if needs_context and conversation_history:
            # Get the last few exchanges for context
            recent_context = "\n".join(conversation_history[-4:]) if len(conversation_history) > 4 else "\n".join(conversation_history)
            
            prompt = f"""Given this conversation context:
{recent_context}

The user now asks: "{user_input.symptoms}"

Rewrite the user's question by replacing pronouns (it, this, that, etc.) with what they're referring to from the context.
If the question is already clear, return it unchanged.

Rewritten question:"""
            
            messages = [
                SystemMessage(content="You are a context resolver that clarifies ambiguous references."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            resolved_input = response.content.strip()
            
            # Update the user input with resolved context
            user_input.symptoms = resolved_input
            
            return {"user_input": user_input, "context_resolved": True}
        
        return {"context_resolved": False}
    
    #=========================================================================
    # TRIAGE AGENT - UPDATED
    #=========================================================================
    
    def _triage_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        conversation_history = state.get("conversation_history", [])
        
        # Include conversation context in triage
        context_str = ""
        if conversation_history:
            context_str = f"\nConversation context:\n" + "\n".join(conversation_history[-2:])
        
        prompt = f"""Classify this input and determine routing:
        Current question: {user_input.symptoms}
        Age: {user_input.age or 'Unknown'}
        {context_str}
        
        Determine:
        1. Intent: [health/non-health]
        2. Urgency: [LOW/MEDIUM/HIGH/EMERGENCY]
        3. Query type:
           - "diet_only": Questions about food, nutrition, supplements, diet
           - "treatment_only": Questions about tests, appointments, when to see doctor
           - "diagnosis_only": Questions about what condition they might have
           - "clarification": Simple follow-ups asking for clarification/details
           - "full_pipeline": New symptoms or complex multi-system issues
        
        Provide your analysis.
        """
        
        messages = [
            SystemMessage(content="You are a medical triage classifier that routes queries efficiently."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.lower()
        
        # Parse response
        intent = "health" if "health" in content else "non-health"
        
        # Determine urgency
        urgency = "MEDIUM"
        if "emergency" in content:
            urgency = "EMERGENCY"
        elif "high" in content:
            urgency = "HIGH"
        elif "low" in content:
            urgency = "LOW"
        
        # Determine routing - always use full pipeline for new/complex cases
        routing = "full_pipeline"
        if "diet_only" in content or any(word in user_input.symptoms.lower() for word in ["diet", "food", "nutrition", "supplement", "vitamin", "eat"]):
            routing = "diet_only"
        elif "treatment_only" in content or any(word in user_input.symptoms.lower() for word in ["test", "appointment", "when should", "blood work", "see doctor"]):
            routing = "treatment_only"
        elif "diagnosis_only" in content or "what is this" in user_input.symptoms.lower():
            routing = "diagnosis_only"
        elif "clarification" in content or (len(user_input.symptoms.split()) < 15 and conversation_history):
            routing = "clarification"
        
        # Override to full pipeline for high urgency
        if urgency in ["HIGH", "EMERGENCY"]:
            routing = "full_pipeline"
        
        triage_result = TriageResult(
            intent_classification=intent,
            urgency_level=urgency,
            emergency_flags=[],
            routing_decision=routing,
            confidence_score=0.8
        )
        
        return {"triage_result": triage_result}
    
    #=========================================================================
    # DIAGNOSIS AGENT - UPDATED
    #=========================================================================
    
    def _diagnosis_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        conversation_history = state.get("conversation_history", [])
        triage_result = state["triage_result"]
        
        # Include context for follow-up questions
        context_str = ""
        if triage_result.routing_decision == "followup" and conversation_history:
            context_str = f"\nThis is a follow-up question. Previous context:\n" + "\n".join(conversation_history[-4:])
        
        # Research symptoms
        wikipedia_results = search_wikipedia.invoke({"query": user_input.symptoms})
        tavily_results = search_tavily.invoke({"query": user_input.symptoms})
        research_content = combine_search_results(wikipedia_results, tavily_results)
        
        # Score severity
        severity_score = score_symptom_severity.invoke({
            "symptoms": [user_input.symptoms], 
            "age": user_input.age or 30
        })
        
        prompt = f"""Analyze symptoms:
        Current question: {user_input.symptoms}
        {context_str}
        Research: {research_content[:1000]}
        Severity: {severity_score}
        
        Provide:
        1. Possible conditions (2-3)
        2. Severity level
        3. Red flags
        4. If this is about blood tests, specify which tests are relevant
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
    # DIET AGENT - UPDATED
    #=========================================================================
    
    def _diet_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        diagnosis_result = state["diagnosis_result"]
        conversation_history = state.get("conversation_history", [])
        
        # Check if this is a diet-related question
        is_diet_related = any(word in user_input.symptoms.lower() for word in ["diet", "food", "eat", "nutrition", "weight"])
        
        if not is_diet_related and conversation_history:
            # Check if previous context was diet-related
            recent_history = " ".join(conversation_history[-2:])
            is_diet_related = any(word in recent_history.lower() for word in ["cholesterol", "weight", "diet", "food"])
        
        # Calculate nutrition needs
        nutrition_calc = calculate_nutrition_needs.invoke({
            "condition": user_input.symptoms,
            "age": user_input.age or 30,
            "gender": user_input.gender or "unknown",
            "activity_level": "moderate"
        })
        
        context_str = ""
        if conversation_history:
            context_str = f"\nConversation context:\n" + "\n".join(conversation_history[-2:])
        
        prompt = f"""Provide dietary recommendations:
        Current question: {user_input.symptoms}
        {context_str}
        Age: {user_input.age or 'Adult'}
        Nutrition needs: {nutrition_calc}
        
        If this is NOT a diet-related question, provide minimal dietary guidance.
        If it IS diet-related, recommend:
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
    # TREATMENT AGENT - UPDATED
    #=========================================================================
    
    def _treatment_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        diagnosis_result = state["diagnosis_result"]
        conversation_history = state.get("conversation_history", [])
        
        # Schedule appointment if needed
        appointment = schedule_appointment.invoke({
            "urgency": diagnosis_result.severity_level.lower(),
            "condition": user_input.symptoms,
            "preferred_timeframe": "soon"
        })
        
        context_str = ""
        if conversation_history:
            context_str = f"\nConversation context:\n" + "\n".join(conversation_history[-2:])
        
        prompt = f"""Provide treatment guidance:
        Current question: {user_input.symptoms}
        {context_str}
        Severity: {diagnosis_result.severity_level}
        Appointment info: {appointment}
        
        If asking about blood tests specifically, detail which tests and why.
        
        Provide:
        1. Self-care measures
        2. When to see doctor
        3. Lifestyle changes
        4. Specific tests if relevant
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
    # SYNTHESIS AGENT - UPDATED
    #=========================================================================
    
    def _synthesis_agent(self, state: MultiAgentHealthState):
        user_input = state["user_input"]
        conversation_history = state.get("conversation_history", [])
        triage_result = state["triage_result"]
        routing = triage_result.routing_decision
        
        # Build context based on which agents were run
        context_parts = []
        
        if state.get("diagnosis_result"):
            context_parts.append(f"Diagnosis: {state['diagnosis_result'].symptom_analysis}")
        
        if state.get("diet_result"):
            context_parts.append("Diet: Nutritional recommendations provided")
        
        if state.get("treatment_result"):
            context_parts.append("Treatment: Care guidance provided")
        
        all_outputs = {}
        if state.get("diagnosis_result"):
            all_outputs["diagnosis"] = state["diagnosis_result"].dict()
        if state.get("diet_result"):
            all_outputs["diet"] = state["diet_result"].dict()
        if state.get("treatment_result"):
            all_outputs["treatment"] = state["treatment_result"].dict()
        
        # Safety validation only if we have recommendations
        safety_check = {"safety_flags": [], "warnings": []}
        if all_outputs:
            safety_check = validate_medical_safety.invoke({
                "recommendations": all_outputs,
                "user_profile": {"age": user_input.age}
            })
        
        context_str = ""
        if conversation_history:
            context_str = f"\nConversation context:\n" + "\n".join(conversation_history[-4:])
        
        # Adjust prompt based on routing
        if routing == "clarification":
            prompt = f"""Provide a direct answer to this clarification:
            Question: {user_input.symptoms}
            {context_str}
            
            Give a concise, direct response.
            """
        elif routing == "diet_only":
            prompt = f"""Create dietary/nutrition response:
            Question: {user_input.symptoms}
            {context_str}
            {' '.join(context_parts)}
            
            Focus only on dietary and nutritional aspects.
            """
        elif routing == "treatment_only":
            prompt = f"""Create treatment/testing response:
            Question: {user_input.symptoms}
            {context_str}
            {' '.join(context_parts)}
            
            Focus only on tests, appointments, and when to see healthcare providers.
            """
        else:
            prompt = f"""Synthesize comprehensive health plan:
            Current question: {user_input.symptoms}
            {context_str}
            {' '.join(context_parts)}
            Safety: {safety_check}
            
            Create unified response that directly answers the user's question.
            """
        
        messages = [
            SystemMessage(content="You are a healthcare coordinator creating focused responses."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        synthesis_result = SynthesisResult(
            all_agent_outputs=all_outputs,
            safety_validations=safety_check.get("safety_flags", []),
            cross_checks={"validated": True},
            final_recommendations={"plan": response.content},
            appointment_needed=state.get("treatment_result", {}).appointment_needed if state.get("treatment_result") else False,
            priority_level=state.get("diagnosis_result", {}).severity_level if state.get("diagnosis_result") else "MEDIUM"
        )
        
        return {"synthesis_result": synthesis_result}
    
    #=========================================================================
    # HALLUCINATION DETECTOR - UNCHANGED
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
    # MAIN PROCESSING - UPDATED
    #=========================================================================
    
    def process_health_query(self, user_input: UserInput, personality: str = "friendly", thread_id: str = "default", progress_callback=None):
        thread = {"configurable": {"thread_id": thread_id}}
        
        # Add current user input to conversation history
        self.conversation_history.append(f"User: {user_input.symptoms}")
        
        # Create a custom graph with progress tracking
        if progress_callback:
            # Process step by step with progress updates
            state = {
                "user_input": user_input,
                "personality": personality,
                "conversation_history": self.conversation_history.copy(),
                "openai_settings": self.openai_settings
            }
            
            # Context Resolver
            progress_callback("🔄 Resolving context...", 0.1)
            state.update(self._context_resolver(state))
            
            # Triage
            progress_callback("🚨 Triaging symptoms...", 0.2)
            state.update(self._triage_agent(state))
            
            # Diagnosis
            progress_callback("🔍 Analyzing symptoms...", 0.3)
            state.update(self._diagnosis_agent(state))
            
            # Diet
            progress_callback("🥗 Generating dietary recommendations...", 0.5)
            state.update(self._diet_agent(state))
            
            # Treatment
            progress_callback("💊 Creating treatment plan...", 0.7)
            state.update(self._treatment_agent(state))
            
            # Synthesis
            progress_callback("🧠 Synthesizing comprehensive plan...", 0.85)
            state.update(self._synthesis_agent(state))
            
            # Validation
            progress_callback("✅ Validating recommendations...", 0.95)
            state.update(self._hallucination_detector(state))
            
            result = state
        else:
            # Normal processing without progress
            result = self.graph.invoke({
                "user_input": user_input,
                "personality": personality,
                "conversation_history": self.conversation_history.copy(),
                "openai_settings": self.openai_settings
            }, thread)
        
        # Generate final response
        synthesis_result = result["synthesis_result"]
        final_plan = synthesis_result.final_recommendations.get("plan", "")
        
        response = f"{final_plan}\n\n{MEDICAL_DISCLAIMER}"
        
        # Add assistant response to conversation history
        self.conversation_history.append(f"Assistant: {final_plan}")
        
        # Keep conversation history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return {
            "response": response,
            "triage_result": result["triage_result"],
            "diagnosis_result": result.get("diagnosis_result", DiagnosisResult(
                symptoms=[],
                symptom_analysis={},
                possible_conditions=[],
                severity_level="MEDIUM",
                recommended_tests=[],
                red_flags=[],
                research_sources=[]
            )),
            "diet_result": result.get("diet_result", DietResult(
                diagnosed_condition="",
                dietary_restrictions=[],
                nutritional_needs={},
                recommended_foods=[],
                foods_to_avoid=[],
                meal_suggestions=[],
                supplements=[]
            )),
            "treatment_result": result.get("treatment_result", TreatmentResult(
                diagnosis={},
                treatment_options=[],
                care_recommendations=[],
                lifestyle_changes=[],
                follow_up_schedule={},
                when_to_see_doctor="",
                appointment_needed=False
            )),
            "synthesis_result": result["synthesis_result"],
            "validation_status": result["hallucination_check"].validation_status
        }