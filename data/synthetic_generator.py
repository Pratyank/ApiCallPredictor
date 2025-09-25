"""
Synthetic Data Generator for OpenSesame Predictor Training
Generates realistic API call prediction training data with diverse patterns and edge cases
"""

import json
import random
import uuid
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generate synthetic training data for the OpenSesame API call predictor
    Creates realistic user prompts, API calls, and confidence scores
    """
    
    def __init__(self):
        self.api_patterns = self._load_api_patterns()
        self.prompt_templates = self._load_prompt_templates()
        self.entity_types = self._load_entity_types()
        self.confidence_ranges = self._load_confidence_ranges()
        
        logger.info("Initialized Synthetic Data Generator for training data creation")
    
    def generate_training_dataset(
        self,
        num_samples: int = 2000,
        include_negative_examples: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive training dataset
        
        Args:
            num_samples: Number of training samples to generate
            include_negative_examples: Include low-quality predictions for contrast
            
        Returns:
            List of training samples with prompts, API calls, and metadata
        """
        
        dataset = []
        
        # Generate positive examples (70%)
        positive_count = int(num_samples * 0.7)
        for _ in range(positive_count):
            sample = self._generate_positive_sample()
            dataset.append(sample)
        
        # Generate neutral examples (20%)
        neutral_count = int(num_samples * 0.2)
        for _ in range(neutral_count):
            sample = self._generate_neutral_sample()
            dataset.append(sample)
        
        # Generate negative examples (10%) if requested
        if include_negative_examples:
            negative_count = num_samples - positive_count - neutral_count
            for _ in range(negative_count):
                sample = self._generate_negative_sample()
                dataset.append(sample)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Generated {len(dataset)} training samples")
        return dataset
    
    def _generate_positive_sample(self) -> Dict[str, Any]:
        """Generate high-quality training sample"""
        
        # Select random API pattern and intent
        pattern = random.choice(list(self.api_patterns.keys()))
        api_info = random.choice(self.api_patterns[pattern])
        
        # Generate realistic user prompt
        prompt = self._generate_prompt_for_api(api_info, pattern, quality="high")
        
        # Create expected API call prediction
        prediction = {
            "api_call": api_info["endpoint"],
            "method": api_info["method"],
            "description": api_info["description"],
            "parameters": api_info.get("parameters", {}),
            "response_format": api_info.get("response_format", "JSON"),
            "confidence": random.uniform(0.75, 0.95)
        }
        
        return {
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "expected_prediction": prediction,
            "intent": pattern,
            "quality_label": "high",
            "metadata": {
                "pattern_type": pattern,
                "generated_at": datetime.utcnow().isoformat(),
                "complexity": "standard"
            }
        }
    
    def _generate_neutral_sample(self) -> Dict[str, Any]:
        """Generate medium-quality training sample"""
        
        pattern = random.choice(list(self.api_patterns.keys()))
        api_info = random.choice(self.api_patterns[pattern])
        
        # Generate somewhat ambiguous prompt
        prompt = self._generate_prompt_for_api(api_info, pattern, quality="medium")
        
        prediction = {
            "api_call": api_info["endpoint"],
            "method": api_info["method"],
            "description": api_info["description"],
            "parameters": api_info.get("parameters", {}),
            "response_format": api_info.get("response_format", "JSON"),
            "confidence": random.uniform(0.4, 0.75)
        }
        
        return {
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "expected_prediction": prediction,
            "intent": pattern,
            "quality_label": "medium",
            "metadata": {
                "pattern_type": pattern,
                "generated_at": datetime.utcnow().isoformat(),
                "complexity": "ambiguous"
            }
        }
    
    def _generate_negative_sample(self) -> Dict[str, Any]:
        """Generate low-quality training sample for contrast learning"""
        
        pattern = random.choice(list(self.api_patterns.keys()))
        api_info = random.choice(self.api_patterns[pattern])
        
        # Generate vague or misleading prompt
        prompt = self._generate_prompt_for_api(api_info, pattern, quality="low")
        
        # Create mismatched or low-confidence prediction
        wrong_pattern = random.choice([p for p in self.api_patterns.keys() if p != pattern])
        wrong_api = random.choice(self.api_patterns[wrong_pattern])
        
        prediction = {
            "api_call": wrong_api["endpoint"],
            "method": wrong_api["method"], 
            "description": wrong_api["description"],
            "parameters": wrong_api.get("parameters", {}),
            "response_format": wrong_api.get("response_format", "JSON"),
            "confidence": random.uniform(0.1, 0.4)
        }
        
        return {
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "expected_prediction": prediction,
            "intent": pattern,
            "quality_label": "low",
            "metadata": {
                "pattern_type": pattern,
                "generated_at": datetime.utcnow().isoformat(),
                "complexity": "mismatch",
                "is_negative_example": True
            }
        }
    
    def _generate_prompt_for_api(self, api_info: Dict[str, Any], pattern: str, quality: str) -> str:
        """Generate realistic user prompt for given API and quality level"""
        
        templates = self.prompt_templates.get(pattern, {}).get(quality, [])
        if not templates:
            templates = ["I need to {action} some {entity}"]
        
        template = random.choice(templates)
        
        # Fill in template variables
        prompt = template.format(
            action=self._get_action_word(api_info["method"]),
            entity=self._get_entity_for_endpoint(api_info["endpoint"]),
            resource=self._get_resource_name(api_info["endpoint"])
        )
        
        # Add context variation
        if quality == "high":
            prompt += f" {self._add_helpful_context(api_info)}"
        elif quality == "low":
            prompt = self._make_prompt_vague(prompt)
        
        return prompt.strip()
    
    def _get_action_word(self, method: str) -> str:
        """Get appropriate action word for HTTP method"""
        action_map = {
            "GET": random.choice(["get", "fetch", "retrieve", "find", "show"]),
            "POST": random.choice(["create", "add", "submit", "post", "make"]),
            "PUT": random.choice(["update", "modify", "change", "replace"]),
            "DELETE": random.choice(["delete", "remove", "destroy", "drop"]),
            "PATCH": random.choice(["update", "modify", "patch", "change"])
        }
        return action_map.get(method.upper(), "access")
    
    def _get_entity_for_endpoint(self, endpoint: str) -> str:
        """Extract entity type from API endpoint"""
        if "users" in endpoint.lower():
            return random.choice(["users", "accounts", "profiles"])
        elif "posts" in endpoint.lower():
            return random.choice(["posts", "articles", "content"])
        elif "orders" in endpoint.lower():
            return random.choice(["orders", "purchases", "transactions"])
        elif "products" in endpoint.lower():
            return random.choice(["products", "items", "inventory"])
        else:
            return random.choice(["data", "resources", "items", "records"])
    
    def _get_resource_name(self, endpoint: str) -> str:
        """Extract resource name from endpoint"""
        parts = endpoint.strip('/').split('/')
        if len(parts) >= 2:
            return parts[-1] if not parts[-1].startswith('{') else parts[-2]
        return "resource"
    
    def _add_helpful_context(self, api_info: Dict[str, Any]) -> str:
        """Add helpful context to make prompt clearer"""
        contexts = [
            f"Please include {', '.join(list(api_info.get('parameters', {}).keys())[:2])} in the response.",
            "Make sure the response is in JSON format.",
            f"This should return {api_info.get('response_format', 'data')}.",
            "I need this for my application integration."
        ]
        return random.choice(contexts)
    
    def _make_prompt_vague(self, prompt: str) -> str:
        """Make prompt intentionally vague or unclear"""
        vague_modifiers = [
            "Maybe I need to",
            "I think I want to",
            "Could you help me",
            "I'm not sure but",
            "Possibly I should"
        ]
        modifier = random.choice(vague_modifiers)
        return f"{modifier} {prompt.lower()}"
    
    def _load_api_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load realistic API patterns for different domains"""
        
        return {
            "user_management": [
                {
                    "endpoint": "/api/users",
                    "method": "GET",
                    "description": "Retrieve list of users",
                    "parameters": {"limit": 10, "offset": 0, "include": "profile"},
                    "response_format": "Array of user objects"
                },
                {
                    "endpoint": "/api/users/{id}",
                    "method": "GET", 
                    "description": "Get specific user by ID",
                    "parameters": {"id": "user_id"},
                    "response_format": "User object"
                },
                {
                    "endpoint": "/api/users",
                    "method": "POST",
                    "description": "Create new user account",
                    "parameters": {"name": "string", "email": "string", "password": "string"},
                    "response_format": "Created user object with ID"
                }
            ],
            
            "content_management": [
                {
                    "endpoint": "/api/posts",
                    "method": "GET",
                    "description": "Retrieve blog posts or articles",
                    "parameters": {"category": "string", "limit": 20},
                    "response_format": "Array of post objects"
                },
                {
                    "endpoint": "/api/posts",
                    "method": "POST",
                    "description": "Create new blog post",
                    "parameters": {"title": "string", "content": "string", "author_id": "string"},
                    "response_format": "Created post object"
                }
            ],
            
            "e_commerce": [
                {
                    "endpoint": "/api/products",
                    "method": "GET",
                    "description": "Browse product catalog",
                    "parameters": {"category": "string", "price_range": "string"},
                    "response_format": "Array of product objects"
                },
                {
                    "endpoint": "/api/orders",
                    "method": "POST",
                    "description": "Place new order",
                    "parameters": {"items": "array", "shipping_address": "object"},
                    "response_format": "Order confirmation object"
                }
            ],
            
            "file_management": [
                {
                    "endpoint": "/api/files",
                    "method": "POST",
                    "description": "Upload file to storage",
                    "parameters": {"file": "binary", "folder": "string"},
                    "response_format": "File metadata object"
                },
                {
                    "endpoint": "/api/files/{id}",
                    "method": "DELETE",
                    "description": "Delete file by ID",
                    "parameters": {"id": "file_id"},
                    "response_format": "Deletion confirmation"
                }
            ],
            
            "search_operations": [
                {
                    "endpoint": "/api/search",
                    "method": "POST",
                    "description": "Search across content and data",
                    "parameters": {"query": "string", "filters": "object", "limit": 50},
                    "response_format": "Search results with metadata"
                }
            ]
        }
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load prompt templates for different quality levels"""
        
        return {
            "user_management": {
                "high": [
                    "I need to {action} user information for user ID {resource}",
                    "Please {action} a new user account with the provided details",
                    "Can you help me {action} the user profile data including {entity}?",
                    "I want to {action} user {entity} from the system database"
                ],
                "medium": [
                    "I need to {action} some {entity} data",
                    "Help me {action} user information",
                    "I want to work with {entity} records"
                ],
                "low": [
                    "Something about {entity}",
                    "I need {entity}",
                    "User stuff"
                ]
            },
            
            "content_management": {
                "high": [
                    "I need to {action} blog posts filtered by category and date",
                    "Please {action} a new article with title, content, and author information",
                    "Can you {action} all published {entity} with their metadata?"
                ],
                "medium": [
                    "I want to {action} some {entity}",
                    "Help me {action} content from the blog"
                ],
                "low": [
                    "Content {action}",
                    "{entity} things"
                ]
            },
            
            "e_commerce": {
                "high": [
                    "I need to {action} products from the catalog with pricing and availability",
                    "Please {action} a new order with items, quantities, and shipping details",
                    "Can you {action} {entity} filtered by category and price range?"
                ],
                "medium": [
                    "I want to {action} {entity} from the store",
                    "Help me {action} some products"
                ],
                "low": [
                    "Shopping {action}",
                    "{entity} buy"
                ]
            }
        }
    
    def _load_entity_types(self) -> List[str]:
        """Load common entity types for variation"""
        return [
            "users", "accounts", "profiles", "customers",
            "posts", "articles", "content", "blogs",
            "products", "items", "inventory", "catalog",
            "orders", "purchases", "transactions", "sales",
            "files", "documents", "images", "media",
            "comments", "reviews", "feedback", "ratings"
        ]
    
    def _load_confidence_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Load confidence score ranges for different quality levels"""
        return {
            "high": (0.75, 0.95),
            "medium": (0.4, 0.75),
            "low": (0.1, 0.4)
        }
    
    def save_dataset_to_file(
        self,
        dataset: List[Dict[str, Any]],
        filename: str = "training_data.json"
    ) -> str:
        """Save generated dataset to JSON file"""
        
        filepath = f"data/training_data/{filename}"
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "metadata": {
                        "generated_at": datetime.utcnow().isoformat(),
                        "total_samples": len(dataset),
                        "generator_version": "1.0.0"
                    },
                    "samples": dataset
                }, f, indent=2)
            
            logger.info(f"Saved {len(dataset)} training samples to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise
    
    def generate_and_save_dataset(
        self,
        num_samples: int = 2000,
        filename: str = None
    ) -> str:
        """Generate and save complete training dataset"""
        
        if filename is None:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dataset = self.generate_training_dataset(num_samples)
        return self.save_dataset_to_file(dataset, filename)

# Convenience functions for quick data generation
def generate_quick_dataset(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Quick dataset generation for testing"""
    generator = SyntheticDataGenerator()
    return generator.generate_training_dataset(num_samples)

def create_training_data_file(num_samples: int = 2000) -> str:
    """Create and save training data file"""
    generator = SyntheticDataGenerator()
    return generator.generate_and_save_dataset(num_samples)

# Example usage for testing
if __name__ == "__main__":
    # Generate sample dataset
    generator = SyntheticDataGenerator()
    sample_data = generator.generate_training_dataset(10)
    
    print("Sample Training Data:")
    for i, sample in enumerate(sample_data[:3]):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {sample['prompt']}")
        print(f"API Call: {sample['expected_prediction']['api_call']}")
        print(f"Confidence: {sample['expected_prediction']['confidence']:.2f}")
        print(f"Quality: {sample['quality_label']}")