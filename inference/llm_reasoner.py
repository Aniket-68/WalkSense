# inference/llm_reasoner.py

import requests
import json
from typing import Optional

class LLMReasoner:
    """
    LLM-based reasoning for query answering and context analysis
    Supports multiple backends: OpenAI API, LM Studio, Ollama
    """
    
    def __init__(self, backend="lm_studio", api_url="http://localhost:1234/v1", model_name="microsoft/phi-4-mini-reasoning"):
        """
        Args:
            backend: "lm_studio", "ollama", or "openai"
            api_url: API endpoint URL
            model_name: Model identifier
        """
        self.backend = backend
        self.api_url = api_url
        self.model_name = model_name
        
        # System prompt for assistive navigation
        self.system_prompt = """You are an AI assistant helping a visually impaired person navigate safely.

Your role:
1. Answer questions about their environment clearly and concisely
2. Provide navigation guidance based on spatial context
3. Alert them to safety concerns
4. Keep responses under 30 words for quick TTS delivery

Be direct, helpful, and safety-focused."""

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

Provide a brief, helpful answer (max 30 words):"""}
        ]
        
        if self.backend == "lm_studio":
            return self._call_lm_studio(messages, max_tokens=100, temperature=0.7)
        elif self.backend == "ollama":
            return self._call_ollama(messages, max_tokens=100, temperature=0.7)
        else:
            return "LLM backend not configured"
    
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
