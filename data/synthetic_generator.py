"""
Synthetic Data Generator for Phase 3 ML Layer - OpenSesame Predictor
Generates 10,000 synthetic SaaS workflow sequences using Markov chains
Stores sequences in data/cache.db using sqlite3
"""

import json
import random
import uuid
import sqlite3
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

# Import database manager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.utils.db_manager import db_manager

logger = logging.getLogger(__name__)

class MarkovChainGenerator:
    """
    Markov chain-based generator for realistic SaaS workflow sequences
    """
    
    def __init__(self):
        self.workflow_transitions = self._build_saas_workflow_transitions()
        self.api_endpoints = self._build_saas_api_endpoints()
        self.workflow_prompts = self._build_workflow_prompts()
        
        # Markov chain parameters
        self.chain_order = 2  # 2-gram transitions for more realistic sequences
        self.max_sequence_length = 15
        self.min_sequence_length = 3
        
        logger.info("Initialized Markov Chain Generator for SaaS workflow sequences")
    
    def _build_saas_workflow_transitions(self) -> Dict[str, Dict[str, float]]:
        """Build transition probabilities for common SaaS workflows"""
        return {
            # Authentication workflows
            'START': {
                'Browse': 0.4,
                'Login': 0.3,
                'Register': 0.2,
                'Search': 0.1
            },
            'Login': {
                'Browse': 0.5,
                'Profile': 0.2,
                'Dashboard': 0.3
            },
            'Register': {
                'Verify': 0.7,
                'Profile': 0.3
            },
            'Verify': {
                'Login': 0.9,
                'Profile': 0.1
            },
            
            # Content management workflows
            'Browse': {
                'View': 0.4,
                'Search': 0.3,
                'Create': 0.2,
                'END': 0.1
            },
            'Search': {
                'Browse': 0.5,
                'View': 0.3,
                'Filter': 0.2
            },
            'Filter': {
                'Browse': 0.6,
                'View': 0.4
            },
            'View': {
                'Edit': 0.3,
                'Delete': 0.1,
                'Share': 0.2,
                'Browse': 0.3,
                'END': 0.1
            },
            'Create': {
                'Edit': 0.6,
                'Save': 0.3,
                'Cancel': 0.1
            },
            'Edit': {
                'Save': 0.7,
                'Preview': 0.2,
                'Cancel': 0.1
            },
            'Preview': {
                'Edit': 0.4,
                'Save': 0.6
            },
            'Save': {
                'Confirm': 0.8,
                'Edit': 0.2
            },
            'Confirm': {
                'View': 0.4,
                'Browse': 0.4,
                'Share': 0.1,
                'END': 0.1
            },
            
            # User management workflows
            'Profile': {
                'Edit': 0.6,
                'Settings': 0.3,
                'Logout': 0.1
            },
            'Settings': {
                'Update': 0.7,
                'Profile': 0.3
            },
            'Update': {
                'Save': 0.8,
                'Cancel': 0.2
            },
            
            # E-commerce workflows
            'Dashboard': {
                'Browse': 0.4,
                'Orders': 0.3,
                'Analytics': 0.2,
                'Settings': 0.1
            },
            'Orders': {
                'View': 0.5,
                'Create': 0.3,
                'Export': 0.2
            },
            'Analytics': {
                'Export': 0.4,
                'Filter': 0.3,
                'Dashboard': 0.3
            },
            
            # File management workflows
            'Upload': {
                'Process': 0.8,
                'Cancel': 0.2
            },
            'Process': {
                'Confirm': 0.9,
                'Retry': 0.1
            },
            'Download': {
                'Confirm': 0.9,
                'Cancel': 0.1
            },
            
            # Sharing workflows
            'Share': {
                'Send': 0.6,
                'Configure': 0.4
            },
            'Configure': {
                'Share': 0.7,
                'Cancel': 0.3
            },
            'Send': {
                'Confirm': 0.8,
                'Edit': 0.2
            },
            
            # Termination states
            'Delete': {
                'Confirm': 0.8,
                'Cancel': 0.2
            },
            'Cancel': {
                'Browse': 0.5,
                'END': 0.5
            },
            'Logout': {
                'END': 1.0
            },
            'Export': {
                'Download': 0.7,
                'END': 0.3
            },
            'Retry': {
                'Process': 0.6,
                'Upload': 0.4
            }
        }
    
    def _build_saas_api_endpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """Map workflow actions to realistic SaaS API endpoints"""
        return {
            'Browse': [
                {'endpoint': '/api/items', 'method': 'GET', 'resource': 'items'},
                {'endpoint': '/api/products', 'method': 'GET', 'resource': 'products'},
                {'endpoint': '/api/documents', 'method': 'GET', 'resource': 'documents'},
                {'endpoint': '/api/users', 'method': 'GET', 'resource': 'users'},
            ],
            'Login': [
                {'endpoint': '/api/auth/login', 'method': 'POST', 'resource': 'auth'},
                {'endpoint': '/api/sessions', 'method': 'POST', 'resource': 'sessions'},
            ],
            'Register': [
                {'endpoint': '/api/auth/register', 'method': 'POST', 'resource': 'auth'},
                {'endpoint': '/api/users', 'method': 'POST', 'resource': 'users'},
            ],
            'Search': [
                {'endpoint': '/api/search', 'method': 'POST', 'resource': 'search'},
                {'endpoint': '/api/items/search', 'method': 'GET', 'resource': 'items'},
            ],
            'View': [
                {'endpoint': '/api/items/{id}', 'method': 'GET', 'resource': 'items'},
                {'endpoint': '/api/documents/{id}', 'method': 'GET', 'resource': 'documents'},
                {'endpoint': '/api/users/{id}', 'method': 'GET', 'resource': 'users'},
            ],
            'Create': [
                {'endpoint': '/api/items', 'method': 'POST', 'resource': 'items'},
                {'endpoint': '/api/documents', 'method': 'POST', 'resource': 'documents'},
                {'endpoint': '/api/projects', 'method': 'POST', 'resource': 'projects'},
            ],
            'Edit': [
                {'endpoint': '/api/items/{id}', 'method': 'PUT', 'resource': 'items'},
                {'endpoint': '/api/documents/{id}', 'method': 'PUT', 'resource': 'documents'},
                {'endpoint': '/api/users/{id}', 'method': 'PATCH', 'resource': 'users'},
            ],
            'Save': [
                {'endpoint': '/api/items/{id}', 'method': 'PUT', 'resource': 'items'},
                {'endpoint': '/api/documents/{id}/save', 'method': 'POST', 'resource': 'documents'},
                {'endpoint': '/api/drafts', 'method': 'POST', 'resource': 'drafts'},
            ],
            'Delete': [
                {'endpoint': '/api/items/{id}', 'method': 'DELETE', 'resource': 'items'},
                {'endpoint': '/api/documents/{id}', 'method': 'DELETE', 'resource': 'documents'},
                {'endpoint': '/api/users/{id}', 'method': 'DELETE', 'resource': 'users'},
            ],
            'Update': [
                {'endpoint': '/api/settings', 'method': 'PUT', 'resource': 'settings'},
                {'endpoint': '/api/profile', 'method': 'PATCH', 'resource': 'profile'},
                {'endpoint': '/api/preferences', 'method': 'PUT', 'resource': 'preferences'},
            ],
            'Confirm': [
                {'endpoint': '/api/actions/confirm', 'method': 'POST', 'resource': 'actions'},
                {'endpoint': '/api/transactions', 'method': 'POST', 'resource': 'transactions'},
            ],
            'Share': [
                {'endpoint': '/api/shares', 'method': 'POST', 'resource': 'shares'},
                {'endpoint': '/api/items/{id}/share', 'method': 'POST', 'resource': 'items'},
            ],
            'Upload': [
                {'endpoint': '/api/files', 'method': 'POST', 'resource': 'files'},
                {'endpoint': '/api/uploads', 'method': 'POST', 'resource': 'uploads'},
            ],
            'Download': [
                {'endpoint': '/api/files/{id}/download', 'method': 'GET', 'resource': 'files'},
                {'endpoint': '/api/exports/{id}', 'method': 'GET', 'resource': 'exports'},
            ]
        }
    
    def _build_workflow_prompts(self) -> Dict[str, List[str]]:
        """Generate realistic user prompts for each workflow action"""
        return {
            'Browse': [
                "I want to see all available items",
                "Show me the product catalog",
                "List all documents in the system",
                "Browse through the available options"
            ],
            'Login': [
                "I need to log into my account",
                "Authenticate me with the system",
                "Sign in to access my data",
                "Login with my credentials"
            ],
            'Search': [
                "Find items matching my criteria",
                "Search for specific products",
                "Look up documents by keyword",
                "Query the database for results"
            ],
            'Create': [
                "I want to create a new item",
                "Add a new document to the system",
                "Start a new project",
                "Generate a fresh entry"
            ],
            'Edit': [
                "Modify the existing item",
                "Update the document content",
                "Change the current settings",
                "Edit the selected record"
            ],
            'Save': [
                "Save my current changes",
                "Store the updated information",
                "Commit the modifications",
                "Persist the current state"
            ],
            'View': [
                "Show me the details",
                "Display the full information",
                "View the complete record",
                "Open the selected item"
            ],
            'Delete': [
                "Remove this item permanently",
                "Delete the selected document",
                "Remove from the system",
                "Permanently erase this record"
            ],
            'Confirm': [
                "Confirm the action",
                "Proceed with the operation",
                "Validate and execute",
                "Approve the changes"
            ]
        }
    
    def generate_workflow_sequence(self) -> Dict[str, Any]:
        """Generate a single workflow sequence using Markov chains"""
        sequence = []
        current_state = 'START'
        sequence_id = str(uuid.uuid4())
        
        # Determine workflow type based on first transition
        workflow_type = self._determine_workflow_type()
        
        for _ in range(self.max_sequence_length):
            if current_state == 'END' or current_state not in self.workflow_transitions:
                break
            
            # Get next state using Markov chain probabilities
            next_states = self.workflow_transitions[current_state]
            next_state = self._weighted_choice(next_states)
            
            if next_state == 'END':
                break
            
            # Generate API call for this state
            if next_state in self.api_endpoints:
                api_info = random.choice(self.api_endpoints[next_state])
                prompt = random.choice(self.workflow_prompts.get(next_state, ["Perform action"]))
                
                step = {
                    'action': next_state,
                    'api_call': api_info['endpoint'],
                    'method': api_info['method'],
                    'resource': api_info['resource'],
                    'prompt': prompt,
                    'timestamp': datetime.now().isoformat(),
                    'step_id': len(sequence) + 1
                }
                sequence.append(step)
            
            current_state = next_state
            
            # Ensure minimum sequence length
            if len(sequence) >= self.min_sequence_length and random.random() < 0.2:
                break
        
        return {
            'sequence_id': sequence_id,
            'workflow_type': workflow_type,
            'sequence_data': sequence,
            'sequence_length': len(sequence),
            'created_at': datetime.now().isoformat()
        }
    
    def _determine_workflow_type(self) -> str:
        """Determine the type of workflow being generated"""
        workflow_types = [
            'content_management', 'user_authentication', 'e_commerce',
            'file_management', 'collaboration', 'analytics', 'settings'
        ]
        return random.choice(workflow_types)
    
    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """Make a weighted random choice based on probabilities"""
        total = sum(choices.values())
        r = random.uniform(0, total)
        upto = 0
        for choice, weight in choices.items():
            if upto + weight >= r:
                return choice
            upto += weight
        return list(choices.keys())[-1]  # Fallback
    
    async def generate_training_sequences(self, num_sequences: int = 10000) -> List[Dict[str, Any]]:
        """Generate large number of training sequences using Markov chains"""
        logger.info(f"Starting generation of {num_sequences} synthetic workflow sequences")
        
        sequences = []
        for i in range(num_sequences):
            sequence = self.generate_workflow_sequence()
            sequences.append(sequence)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_sequences} sequences")
        
        logger.info(f"Completed generation of {len(sequences)} workflow sequences")
        return sequences
    
    async def generate_positive_negative_examples(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate positive and negative training examples from sequences"""
        training_examples = []
        
        for sequence in sequences:
            sequence_data = sequence['sequence_data']
            
            # Generate positive examples (actual next calls)
            for i, step in enumerate(sequence_data[:-1]):
                next_step = sequence_data[i + 1]
                
                positive_example = {
                    'sequence_id': sequence['sequence_id'],
                    'prompt': step['prompt'],
                    'api_call': next_step['api_call'],
                    'method': next_step['method'],
                    'is_positive': True,
                    'rank': 1,
                    'context_position': i,
                    'features': self._extract_sequence_features(sequence_data, i)
                }
                training_examples.append(positive_example)
            
            # Generate negative examples (random sampling from other sequences)
            num_negatives = min(3, len(sequence_data) - 1)  # 3:1 negative:positive ratio
            for i in range(num_negatives):
                # Random step from current sequence
                current_step = random.choice(sequence_data[:-1])
                
                # Random API call from different sequence
                other_sequence = random.choice(sequences)
                other_step = random.choice(other_sequence['sequence_data'])
                
                negative_example = {
                    'sequence_id': sequence['sequence_id'],
                    'prompt': current_step['prompt'],
                    'api_call': other_step['api_call'],
                    'method': other_step['method'],
                    'is_positive': False,
                    'rank': random.randint(2, 5),
                    'context_position': -1,
                    'features': {}
                }
                training_examples.append(negative_example)
        
        logger.info(f"Generated {len(training_examples)} training examples (positive + negative)")
        return training_examples
    
    def _extract_sequence_features(self, sequence: List[Dict[str, Any]], position: int) -> Dict[str, Any]:
        """Extract basic features from sequence context"""
        if position == 0:
            return {'is_first_step': True}
        
        previous_step = sequence[position - 1]
        return {
            'previous_action': previous_step['action'],
            'previous_resource': previous_step['resource'],
            'previous_method': previous_step['method'],
            'position_in_sequence': position,
            'sequence_progress': position / len(sequence)
        }


class SyntheticDataGenerator:
    """
    Main generator class for Phase 3 ML Layer synthetic data
    """
    
    def __init__(self):
        self.markov_generator = MarkovChainGenerator()
        self.db_manager = db_manager
    
    async def generate_and_store_sequences(self, num_sequences: int = 10000) -> Dict[str, Any]:
        """Generate and store 10,000 synthetic sequences in database"""
        start_time = datetime.now()
        
        # Generate sequences using Markov chains
        sequences = await self.markov_generator.generate_training_sequences(num_sequences)
        
        # Store sequences in database
        stored_count = await self.db_manager.store_synthetic_sequences(sequences)
        
        # Generate positive and negative training examples
        training_examples = await self.markov_generator.generate_positive_negative_examples(sequences)
        
        # Store training examples
        training_stored = await self.db_manager.store_training_data(training_examples)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats = {
            'sequences_generated': len(sequences),
            'sequences_stored': stored_count,
            'training_examples': len(training_examples),
            'training_examples_stored': training_stored,
            'processing_time_seconds': processing_time,
            'average_sequence_length': sum(s['sequence_length'] for s in sequences) / len(sequences),
            'workflow_types': list(set(s['workflow_type'] for s in sequences)),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Generation complete: {stats}")
        return stats
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated data"""
        db_stats = await self.db_manager.get_database_stats()
        return {
            'database_stats': db_stats,
            'markov_chain_order': self.markov_generator.chain_order,
            'max_sequence_length': self.markov_generator.max_sequence_length,
            'min_sequence_length': self.markov_generator.min_sequence_length,
            'workflow_types_available': len(self.markov_generator.workflow_transitions),
            'api_endpoints_available': sum(len(endpoints) for endpoints in self.markov_generator.api_endpoints.values())
        }


# Convenience functions for Phase 3 integration
async def generate_ml_training_data(num_sequences: int = 10000) -> Dict[str, Any]:
    """Generate ML training data for Phase 3"""
    generator = SyntheticDataGenerator()
    return await generator.generate_and_store_sequences(num_sequences)

async def get_training_data_stats() -> Dict[str, Any]:
    """Get training data generation statistics"""
    generator = SyntheticDataGenerator()
    return await generator.get_generation_stats()


# Example usage
if __name__ == "__main__":
    async def main():
        # Generate 10,000 synthetic SaaS workflow sequences
        generator = SyntheticDataGenerator()
        stats = await generator.generate_and_store_sequences(1000)  # Use 1000 for testing
        
        print("Generation Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show sample sequences
        sequences = await db_manager.get_synthetic_sequences(5)
        print(f"\nSample sequences (first 5):")
        for i, seq in enumerate(sequences):
            print(f"\nSequence {i+1} ({seq['workflow_type']}):")
            for step in seq['sequence_data'][:3]:  # Show first 3 steps
                print(f"  - {step['action']}: {step['api_call']} ({step['method']})")
    
    asyncio.run(main())