    def _init_ollama(self):
        """Initialize Ollama backend"""
        print(f"[QWEN] Using Ollama backend")
        print(f"[QWEN] API URL: {self.ollama_url}")
        print(f"[QWEN] Model: {self.model_id}")
        
        # Test connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m['name'] for m in models_data.get('models', [])]
                print(f"[QWEN] Connected to Ollama successfully")
                print(f"[QWEN] Available models: {', '.join(available_models)}")
                
                if self.model_id in available_models:
                    print(f"[QWEN] ✓ Model '{self.model_id}' is available")
                else:
                    print(f"[QWEN WARNING] Model '{self.model_id}' not found!")
                    print(f"[QWEN] Run: ollama pull {self.model_id}")
                    
            else:
                print(f"[QWEN WARNING] Ollama responded with status {response.status_code}")
                
        except Exception as e:
            print(f"[QWEN WARNING] Could not connect to Ollama: {e}")
            print("[QWEN] Make sure Ollama is running!")
            print("[QWEN] Start with: ollama serve")

    def describe_scene_ollama(self, frame, context=""):
        """Use Ollama API for scene description"""
        try:
            # Encode image to base64
            image_base64 = self._encode_image_base64(frame)
            
            # Create prompt
            prompt = "Describe this scene briefly for a visually impaired person. Focus on obstacles, people, and navigation hazards. Keep it under 30 words."
            if context:
                prompt = f"Object Detections: {context}. {prompt}"
            
            # Prepare request for Ollama
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "images": [image_base64.split(",")[1]],  # Remove data:image/jpeg;base64, prefix
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100
                }
            }
            
            # Make API call
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '').strip()
                
                if description:
                    return description
                else:
                    return "No description available"
            else:
                print(f"[QWEN ERROR] Ollama returned status {response.status_code}")
                return "Vision model error"
                
        except Exception as e:
            print(f"[QWEN ERROR] Ollama scene description failed: {e}")
            return f"Error: {str(e)}"
