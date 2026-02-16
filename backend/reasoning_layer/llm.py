# inference/llm_reasoner.py

import requests
import json
import os
from typing import Optional

class LLMReasoner:
    """
    LLM-based reasoning for query answering and context analysis
    Supports multiple backends: OpenAI API, LM Studio, Ollama
    """
    
    def __init__(self, backend="lm_studio", api_url="http://localhost:1234/v1", model_name="microsoft/phi-4-mini-reasoning", api_key=None):
        """
        Args:
            backend: "lm_studio", "ollama", "gemini", or "openai"
            api_url: API endpoint URL
            model_name: Model identifier
            api_key: API key for Gemini (optional, can use env var)
        """
        self.backend = backend
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize Gemini if needed
        if self.backend == "gemini":
            try:
                import google.generativeai as genai
                key = self.api_key or os.getenv("GEMINI_API_KEY")
                if not key:
                    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY env var or pass api_key parameter.")
                genai.configure(api_key=key)
                self.gemini_model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        from loguru import logger
        logger.info(f"Connected to Backend: '{self.backend}' | Model: '{self.model_name}'")
        
        # System prompt for visual synthesis (Query-Focused)
        # System prompt for visual synthesis (Jarvis Style)
        self.system_prompt = """You are 'WalkSense AI', a helpful and concise visual assistant.
Your task: Use 'VLM Observations' and 'Spatial Context' to answer the User's questions.

GUIDELINES:
1. Answer directly. DO NOT repeat the user's question.
2. Be natural, like Jarvis. Don't be overly technical.
3. ALWAYS prioritize visual proof. If you don't see it, politely say so.
4. Keep responses brief (under 25 words) and actionable."""

    def check_health(self):
        """
        Verify connection to the backend
        """
        try:
            if self.backend == "ollama":
                resp = requests.get(f"{self.api_url.replace('/api/generate', '')}/", timeout=1) # Ollama root
                return resp.status_code == 200
            else:
                resp = requests.get(f"{self.api_url}/models", timeout=1)
                return resp.status_code == 200
        except:
            return False

    def _call_lm_studio(self, messages, max_tokens=100, temperature=0.7):
        """Call LM Studio API"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def _call_ollama(self, messages, max_tokens=100, temperature=0.7):
        """Call Ollama API"""
        try:
            # Convert messages to Ollama format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def _call_gemini(self, messages, max_tokens=100, temperature=0.7):
        """Call Gemini API"""
        try:
            # Convert messages to Gemini format
            # Gemini expects a single prompt or conversation history
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            
            # Combine system and user messages
            full_prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
            
            # Generate response
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens,
                }
            )
            
            return response.text.strip()
                
        except Exception as e:
            return f"LLM Error: {str(e)}"

    
    def answer_query(self, 
                     user_query: str,
                     spatial_context: str,
                     scene_description: Optional[str] = None) -> str:
        """
        Answer user query using spatial context and scene understanding
        
        Args:
            user_query: User's question (from STT)
            spatial_context: Current spatial state from SpatialContextManager
            scene_description: Latest VLM scene description
            
        Returns:
            LLM-generated answer
        """
        # Build context-aware prompt
        context_parts = [spatial_context]
        if scene_description:
            context_parts.append(f"\nVLM Description: {scene_description}")
        
        full_context = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Context:
{full_context}

User Question: {user_query}

Provide a brief, helpful answer (max 30 words). DO NOT repeat the question:"""}
        ]
        
        if self.backend == "lm_studio":
            print(f"[LLM DEBUG] Processing Query via LM Studio: '{user_query}'")
            response = self._call_lm_studio(messages, max_tokens=100, temperature=0.7)
        elif self.backend == "ollama":
            print(f"[LLM DEBUG] Processing Query via Ollama: '{user_query}'")
            response = self._call_ollama(messages, max_tokens=100, temperature=0.7)
        elif self.backend == "gemini":
            print(f"[LLM DEBUG] Processing Query via Gemini: '{user_query}'")
            response = self._call_gemini(messages, max_tokens=100, temperature=0.7)
        else:
            return "LLM backend not configured"
            
        # Post-processing: remove query repetition if present
        if response and response.lower().startswith(user_query.lower()):
            response = response[len(user_query):].strip()
            # Remove leading punctuation like "?" or ":" or "-"
            response = response.lstrip("?:- ")
            
        return response
    
    def analyze_safety(self, spatial_context: str, scene_description: str) -> Optional[str]:
        """
        Analyze scene for unreported safety concerns
        
        Returns:
            Safety alert message or None
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Analyze this scene for safety concerns:

Spatial Context:
{spatial_context}

Scene Description:
{scene_description}

Are there any safety hazards not already mentioned? If yes, provide a brief warning (15 words max).
If no additional hazards, respond with "SAFE"."""}
        ]
        
        if self.backend == "lm_studio":
            response = self._call_lm_studio(messages, max_tokens=50, temperature=0.3)
        elif self.backend == "ollama":
            response = self._call_ollama(messages, max_tokens=50, temperature=0.3)
        else:
            return None
        
        if response and response.strip().upper() != "SAFE":
            return response
        return None
    

    def generate_navigation_hint(self, spatial_context: str) -> str:
        """
        Generate proactive navigation guidance
        
        Returns:
            Navigation hint based on current spatial state
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Based on this environment, provide ONE brief navigation tip (max 20 words):

{spatial_context}

Tip:"""}
        ]
        
        if self.backend == "lm_studio":
            return self._call_lm_studio(messages, max_tokens=60, temperature=0.8)
        elif self.backend == "ollama":
            return self._call_ollama(messages, max_tokens=60, temperature=0.8)
        else:
            return "Clear path ahead"
