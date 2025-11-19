"""
Gemini API Integration Module
Handles conversation with Google Gemini for requirement extraction
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import json
import re


class GeminiAgent:
    """Intelligent agent using Gemini API to extract bungalow requirements"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize Gemini Agent
        
        Args:
            api_key: Google Gemini API key
            model_name: Model to use (default: gemini-pro)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = None
        self.conversation_history = []
        self.extracted_requirements = {}
        
    def start_conversation(self) -> str:
        """Start a new conversation session"""
        system_prompt = """You are a professional architectural consultant helping customers design their dream bungalow. 
Your goal is to extract detailed requirements through natural, friendly conversation.

You need to gather:
1. Bungalow type (single-story, duplex, villa, cottage)
2. Land area (in sq ft or sq meters)
3. Number of bedrooms (and preferred sizes)
4. Number of bathrooms
5. Living room requirement (yes/no, size)
6. Kitchen type (open/closed/semi-open)
7. Dining room (separate/combined with living)
8. Parking (2-wheeler/4-wheeler capacity)
9. Additional rooms (study, guest room, prayer room, store room)
10. Balconies/terraces
11. Special preferences (open floor plan, room adjacencies, natural light, entrance direction)
12. Garden or outdoor space requirements

Ask questions one at a time or in small logical groups. Be conversational and friendly. 
If the user is unsure, provide examples or suggestions. Keep track of what's been answered.
After gathering sufficient information, summarize and confirm."""

        self.chat = self.model.start_chat(history=[])
        
        welcome_message = self.chat.send_message(
            f"{system_prompt}\n\nNow greet the customer and start gathering requirements. Begin with a warm welcome."
        )
        
        response = welcome_message.text
        self.conversation_history.append({"role": "agent", "content": response})
        return response
    
    def send_message(self, user_input: str) -> str:
        """
        Send user message and get agent response
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = self.chat.send_message(user_input)
        agent_response = response.text
        
        self.conversation_history.append({"role": "agent", "content": agent_response})
        return agent_response
    
    def extract_requirements_from_conversation(self) -> Dict:
        """
        Extract structured requirements from conversation history
        
        Returns:
            Dictionary with extracted requirements
        """
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        extraction_prompt = f"""Based on the following conversation, extract all bungalow requirements into a structured JSON format.

Conversation:
{conversation_text}

Extract into this JSON structure (use null for missing information):
{{
    "bungalow_type": "single-story/duplex/villa/cottage",
    "land_area": {{"value": number, "unit": "sqft/sqm"}},
    "bedrooms": {{"count": number, "sizes": ["string describing size preferences"]}},
    "bathrooms": {{"count": number, "types": ["master/common/powder"]}},
    "living_room": {{"required": true/false, "size": "small/medium/large/custom"}},
    "kitchen": {{"type": "open/closed/semi-open", "size": "string"}},
    "dining_room": {{"type": "separate/combined", "location": "string"}},
    "parking": {{"two_wheeler": number, "four_wheeler": number}},
    "additional_rooms": {{"study": bool, "guest_room": bool, "prayer_room": bool, "store_room": bool, "laundry": bool}},
    "outdoor_spaces": {{"balcony": bool, "terrace": bool, "garden": bool, "patio": bool}},
    "preferences": {{
        "floor_plan_style": "open/traditional",
        "natural_light": "high/medium/low",
        "entrance_direction": "north/south/east/west/flexible",
        "room_adjacencies": ["bedroom near bathroom", etc],
        "special_requirements": ["accessibility features", etc]
    }}
}}

Return ONLY valid JSON, no additional text."""

        response = self.model.generate_content(extraction_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
                self.extracted_requirements = requirements
                return requirements
            else:
                return {"error": "Could not extract JSON from response"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": response.text}
    
    def get_conversation_summary(self) -> str:
        """Generate a human-readable summary of requirements"""
        if not self.extracted_requirements:
            return "No requirements extracted yet."
        
        req = self.extracted_requirements
        summary_parts = []
        
        summary_parts.append("=== BUNGALOW DESIGN REQUIREMENTS ===\n")
        
        if req.get("bungalow_type"):
            summary_parts.append(f"🏠 Type: {req['bungalow_type'].title()}")
        
        if req.get("land_area"):
            land = req["land_area"]
            summary_parts.append(f"📏 Land Area: {land.get('value')} {land.get('unit')}")
        
        if req.get("bedrooms"):
            br = req["bedrooms"]
            summary_parts.append(f"🛏️  Bedrooms: {br.get('count')}")
        
        if req.get("bathrooms"):
            summary_parts.append(f"🚿 Bathrooms: {req['bathrooms'].get('count')}")
        
        if req.get("living_room", {}).get("required"):
            summary_parts.append(f"🛋️  Living Room: {req['living_room'].get('size', 'Yes')}")
        
        if req.get("kitchen"):
            summary_parts.append(f"🍳 Kitchen: {req['kitchen'].get('type', 'Standard')}")
        
        if req.get("dining_room"):
            summary_parts.append(f"🍽️  Dining: {req['dining_room'].get('type', 'Standard')}")
        
        if req.get("parking"):
            parking = req["parking"]
            summary_parts.append(f"🚗 Parking: {parking.get('two_wheeler', 0)} bikes, {parking.get('four_wheeler', 0)} cars")
        
        additional = req.get("additional_rooms", {})
        add_rooms = [room.replace('_', ' ').title() for room, needed in additional.items() if needed]
        if add_rooms:
            summary_parts.append(f"➕ Additional: {', '.join(add_rooms)}")
        
        outdoor = req.get("outdoor_spaces", {})
        outdoor_list = [space.title() for space, needed in outdoor.items() if needed]
        if outdoor_list:
            summary_parts.append(f"🌳 Outdoor: {', '.join(outdoor_list)}")
        
        preferences = req.get("preferences", {})
        if preferences:
            summary_parts.append("\n📝 Preferences:")
            if preferences.get("floor_plan_style"):
                summary_parts.append(f"   - Style: {preferences['floor_plan_style'].title()}")
            if preferences.get("natural_light"):
                summary_parts.append(f"   - Natural Light: {preferences['natural_light'].title()}")
            if preferences.get("entrance_direction"):
                summary_parts.append(f"   - Entrance: {preferences['entrance_direction'].title()}")
        
        return "\n".join(summary_parts)
    
    def save_requirements(self, filepath: str):
        """Save extracted requirements to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.extracted_requirements, indent=2, fp=f)
    
    def load_requirements(self, filepath: str):
        """Load requirements from JSON file"""
        with open(filepath, 'r') as f:
            self.extracted_requirements = json.load(f)
