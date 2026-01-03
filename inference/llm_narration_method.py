    
    def narrate_environment(self, detections: list, max_words: int = 15) -> str:
        """
        Generate ONE concise navigation line from YOLO detections
        
        Args:
            detections: List of YOLO detection dicts with 'label', 'confidence', 'bbox'
            max_words: Maximum words in output
            
        Returns:
            Single-line navigation guidance (e.g., "Person ahead, car on right")
        """
        if not detections:
            return ""
        
        # Filter for important navigation objects only
        important_objects = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
        relevant_dets = [d for d in detections if d['label'] in important_objects]
        
        if not relevant_dets:
            return ""  # Nothing important to announce
        
        # Create simple detection summary (max 3 most important)
        detection_summary = []
        for det in relevant_dets[:3]:
            label = det['label']
            # Simple direction from bbox x-coordinate (normalized 0-1)
            bbox = det['bbox'][0]  # [x1, y1, x2, y2]
            # Assuming bbox coordinates are in pixels, normalize them
            # For now, use center x position
            detection_summary.append(label)
        
        detections_text = ", ".join(detection_summary)
        
        # Ask LLM to create ONE concise navigation line
        messages = [
            {"role": "system", "content": "You are a GPS-style navigation assistant. Be extremely concise."},
            {"role": "user", "content": f"""Detected: {detections_text}

ONE sentence ({max_words} words max), GPS-style:"""}
        ]
        
        if self.backend == "lm_studio":
            result = self._call_lm_studio(messages, max_tokens=40, temperature=0.3)
        elif self.backend == "ollama":
            result = self._call_ollama(messages, max_tokens=40, temperature=0.3)
        else:
            # Simple fallback
            if relevant_dets:
                return f"{relevant_dets[0]['label']} detected"
            return ""
        
        return result if result and not result.startswith("LLM Error") else ""
