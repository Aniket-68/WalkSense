# reasoning/qwen_vlm.py

import torch
import cv2
import os
import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVLM:
    """
    Qwen Vision-Language Model with multiple backend support:
    - 'lm_studio': Use LM Studio local API server
    - 'huggingface': Use downloaded HuggingFace model
    """
    
    def __init__(self, backend=None, model_id=None, lm_studio_url=None):
        """
        Initialization pulls from the Central Registry if arguments are not provided.
        """
        from utils.config_loader import Config
        self.backend = backend or Config.get("vlm.active_provider", "lm_studio")
        
        # Get provider-specific settings
        provider_config = f"vlm.providers.{self.backend}"
        self.url = lm_studio_url or Config.get(f"{provider_config}.url")
        self.model_id = model_id or Config.get(f"{provider_config}.model_id")
        
        if self.backend == "lm_studio":
            self._init_lm_studio()
        elif self.backend == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _init_huggingface(self):
        """Initialize local HuggingFace model"""
        print(f"[QWEN] Loading local model: {self.model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Processor and Model
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype="auto", 
            device_map="auto"
        )
        print(f"[QWEN] Model loaded on {self.device}")

    def _init_lm_studio(self):
        """Initialize LM Studio backend"""
        print(f"[QWEN] Using LM Studio backend: {self.url}")
        
        try:
            response = requests.get(f"{self.url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models.get('data'):
                    self.active_model_id = models['data'][0]['id']
                    print(f"[QWEN] Connected. Active: {self.active_model_id}")
                else:
                    self.active_model_id = self.model_id
            else:
                self.active_model_id = self.model_id
        except Exception as e:
            print(f"[QWEN WARNING] API offline: {e}")
            self.active_model_id = self.model_id

    def _encode_image_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"

    def describe_scene_lm_studio(self, frame, context=""):
        """Use LM Studio API for scene description"""
        try:
            # Encode image
            image_base64 = self._encode_image_base64(frame)
            
            # Create fused prompt
            prompt = "Describe this scene briefly for a visually impaired person. Focus on obstacles, people, and navigation hazards. Keep it under 30 words."
            if context:
                prompt = f"Object Detections: {context}. {prompt}"
            
            # Prepare request
            payload = {
                "model": self.active_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            # Make request
            response = requests.post(
                f"{self.url}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content']
                return description.strip()
            else:
                return f"LM Studio error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def describe_scene_huggingface(self, frame, context=""):
        """Use HuggingFace model for scene description"""
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Create fused prompt
        prompt = "Describe this scene briefly for a visually impaired person. Focus on obstacles, people, and navigation hazards."
        if context:
            prompt = f"Known objects: {context}. {prompt}"
        
        # Prepare messages in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def describe_scene(self, frame, context=""):
        """
        Main method to describe scene - routes to appropriate backend
        
        Args:
            frame: OpenCV BGR frame (numpy array)
            context: Text string containing known objects/detections
        Returns:
            str: Scene description
        """
        if self.backend == "lm_studio":
            return self.describe_scene_lm_studio(frame, context)
        else:
            return self.describe_scene_huggingface(frame, context)
