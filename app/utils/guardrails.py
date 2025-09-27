import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta

from app.config import get_settings

logger = logging.getLogger(__name__)

# Phase 4: Destructive Patterns Detection
DESTRUCTIVE_PATTERNS = {
    'DELETE': lambda endpoint, params: True,  # All DELETE operations are potentially destructive
    'PUT': lambda endpoint, params: 'delete' in endpoint.lower() or 'remove' in endpoint.lower(),
    'PATCH': lambda endpoint, params: check_critical_fields(params)
}

def check_critical_fields(params: Dict[str, Any]) -> bool:
    """
    Check if parameters contain critical fields that could be destructive
    
    Args:
        params: Dictionary of API parameters
        
    Returns:
        True if critical fields are detected, False otherwise
    """
    if not params or not isinstance(params, dict):
        return False
    
    # Critical field patterns that indicate potential destruction
    critical_patterns = [
        'delete', 'remove', 'drop', 'truncate', 'clear', 'destroy',
        'purge', 'erase', 'wipe', 'reset', 'terminate', 'kill'
    ]
    
    # Check parameter keys and values
    for key, value in params.items():
        key_lower = str(key).lower()
        value_str = str(value).lower()
        
        # Check if key or value contains critical patterns
        for pattern in critical_patterns:
            if pattern in key_lower or pattern in value_str:
                return True
    
    return False

def is_safe(endpoint: str, params: Dict[str, Any] = None, prompt: str = None) -> Tuple[bool, str]:
    """
    Enhanced safety check for API endpoints with destructive pattern detection
    
    Args:
        endpoint: API endpoint to check
        params: Dictionary of API parameters
        prompt: Original user prompt for intent analysis
        
    Returns:
        Tuple of (is_safe, reason) where is_safe is boolean and reason is explanation
    """
    if not endpoint:
        return False, "Empty endpoint provided"
    
    # Extract HTTP method from endpoint or assume GET
    parts = endpoint.split(' ', 1)
    if len(parts) == 2:
        method, path = parts
        method = method.upper()
    else:
        method = 'GET'
        path = endpoint
    
    # Check destructive patterns based on HTTP method
    if method in DESTRUCTIVE_PATTERNS:
        pattern_checker = DESTRUCTIVE_PATTERNS[method]
        if pattern_checker(path, params or {}):
            return False, f"Destructive pattern detected for {method} operation"
    
    # Additional safety checks for specific patterns
    
    # Check for bulk operations
    if any(keyword in path.lower() for keyword in ['bulk', 'batch', 'mass']):
        if method in ['DELETE', 'PUT', 'PATCH']:
            return False, "Bulk destructive operation detected"
    
    # Check for admin/system endpoints
    if any(keyword in path.lower() for keyword in ['admin', 'system', 'root', 'config']):
        if method in ['DELETE', 'PUT', 'PATCH']:
            return False, "Administrative destructive operation detected"
    
    # Check for database-related destructive operations
    db_destructive_patterns = ['drop', 'truncate', 'delete_all', 'clear_all']
    if any(pattern in path.lower() for pattern in db_destructive_patterns):
        return False, "Database destructive operation detected"
    
    # Prompt intent analysis if provided
    if prompt:
        prompt_lower = prompt.lower()
        destructive_intents = [
            'delete all', 'remove everything', 'clear all data', 'wipe clean',
            'destroy', 'purge', 'reset everything', 'drop table', 'truncate'
        ]
        
        for intent in destructive_intents:
            if intent in prompt_lower:
                return False, f"Destructive intent detected in prompt: '{intent}'"
    
    # Check parameter safety
    if params:
        # Check for wildcard parameters that could affect multiple records
        for key, value in params.items():
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ['*', 'all', '%', '%%'] and method in ['DELETE', 'PUT', 'PATCH']:
                    return False, f"Wildcard parameter '{key}={value}' with destructive method"
                
                # Check for SQL-like destructive patterns in parameters
                if any(pattern in value_lower for pattern in ['drop table', 'delete from', 'truncate']):
                    return False, f"SQL destructive pattern in parameter: {key}"
    
    return True, "Operation appears safe"

class SafetyValidator:
    """
    Safety validation and guardrails system for API prediction inputs and outputs
    Implements content filtering, rate limiting, and security checks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.max_prompt_length = self.settings.max_prompt_length
        self.blocked_patterns = self._load_blocked_patterns()
        self.rate_limits = {}
        
        # Security patterns
        self.sql_injection_patterns = [
            r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table)",
            r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
            r"(?i)(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
        ]
        
        self.xss_patterns = [
            r"(?i)(<script|</script>|javascript:|on\w+\s*=)",
            r"(?i)(eval\s*\(|settimeout\s*\(|setinterval\s*\()",
            r"(?i)(document\.|window\.|alert\s*\()",
        ]
        
        self.path_traversal_patterns = [
            r"(\.\./|\.\.\|/\.\.)",
            r"(?i)(etc/passwd|etc/shadow|boot\.ini)",
            r"(?i)(cmd\.exe|powershell|/bin/sh)",
        ]
        
        # Validation statistics
        self.total_validations = 0
        self.blocked_requests = 0
        self.security_violations = 0
        
        logger.info("Initialized Safety Validator with content filtering and security checks")
    
    def validate_input(
        self, 
        prompt: str, 
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Comprehensive input validation with safety checks
        
        Args:
            prompt: User input prompt to validate
            user_id: Optional user identifier for rate limiting
            additional_context: Additional request context
            
        Returns:
            True if input is safe, False if blocked
        """
        
        self.total_validations += 1
        
        try:
            # Basic length validation
            if not self._validate_length(prompt):
                logger.warning(f"Prompt length validation failed: {len(prompt)} chars")
                self.blocked_requests += 1
                return False
            
            # Content safety checks
            if not self._validate_content_safety(prompt):
                logger.warning("Content safety validation failed")
                self.blocked_requests += 1
                return False
            
            # Security vulnerability checks
            if not self._validate_security(prompt):
                logger.warning("Security validation failed")
                self.security_violations += 1
                self.blocked_requests += 1
                return False
            
            # Rate limiting checks
            if user_id and not self._check_rate_limit(user_id):
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                self.blocked_requests += 1
                return False
            
            # Additional context validation
            if additional_context and not self._validate_additional_context(additional_context):
                logger.warning("Additional context validation failed")
                self.blocked_requests += 1
                return False
            
            logger.debug("Input validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            # Fail closed - block on error
            self.blocked_requests += 1
            return False
    
    def validate_output(
        self, 
        predictions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate and filter API predictions for safety
        
        Args:
            predictions: List of API call predictions to validate
            
        Returns:
            Tuple of (filtered_predictions, warnings)
        """
        
        try:
            filtered_predictions = []
            warnings = []
            
            for i, prediction in enumerate(predictions):
                # Validate prediction structure
                if not self._validate_prediction_structure(prediction):
                    warnings.append(f"Prediction {i}: Invalid structure")
                    continue
                
                # Security checks for API calls
                api_call = prediction.get("api_call", "")
                if not self._validate_api_call_security(api_call):
                    warnings.append(f"Prediction {i}: Security check failed for API call")
                    continue
                
                # Parameter validation
                params = prediction.get("parameters", {})
                if not self._validate_parameters(params):
                    warnings.append(f"Prediction {i}: Invalid parameters")
                    # Clean parameters instead of dropping prediction
                    prediction["parameters"] = self._sanitize_parameters(params)
                
                # Content filtering
                description = prediction.get("description", "")
                if not self._validate_description_content(description):
                    warnings.append(f"Prediction {i}: Description content filtered")
                    prediction["description"] = self._sanitize_description(description)
                
                filtered_predictions.append(prediction)
            
            logger.info(f"Validated {len(predictions)} predictions, filtered to {len(filtered_predictions)}")
            return filtered_predictions, warnings
            
        except Exception as e:
            logger.error(f"Output validation error: {str(e)}")
            return [], [f"Validation error: {str(e)}"]
    
    def _validate_length(self, prompt: str) -> bool:
        """Validate prompt length constraints"""
        return 1 <= len(prompt) <= self.max_prompt_length
    
    def _validate_content_safety(self, prompt: str) -> bool:
        """Check for inappropriate content using blocked patterns"""
        
        prompt_lower = prompt.lower()
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt_lower):
                logger.debug(f"Blocked pattern matched: {pattern}")
                return False
        
        # Check for excessive profanity or offensive content
        if self._contains_excessive_profanity(prompt):
            return False
        
        # Check for personal information patterns
        if self._contains_personal_info(prompt):
            logger.debug("Personal information detected in prompt")
            return False
        
        return True
    
    def _validate_security(self, prompt: str) -> bool:
        """Security vulnerability checks"""
        
        # SQL Injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, prompt):
                logger.warning("SQL injection pattern detected")
                return False
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, prompt):
                logger.warning("XSS pattern detected")
                return False
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, prompt):
                logger.warning("Path traversal pattern detected")
                return False
        
        # Command injection detection
        if self._contains_command_injection(prompt):
            logger.warning("Command injection pattern detected")
            return False
        
        return True
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check rate limiting for user"""
        
        current_time = datetime.utcnow()
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        
        # Initialize user rate limit tracking
        if user_hash not in self.rate_limits:
            self.rate_limits[user_hash] = {
                "requests": [],
                "blocked_until": None
            }
        
        user_limits = self.rate_limits[user_hash]
        
        # Check if user is currently blocked
        if user_limits["blocked_until"] and current_time < user_limits["blocked_until"]:
            return False
        
        # Clean old requests (1-minute window)
        cutoff_time = current_time - timedelta(minutes=1)
        user_limits["requests"] = [
            req_time for req_time in user_limits["requests"] 
            if req_time > cutoff_time
        ]
        
        # Check rate limit (60 requests per minute)
        if len(user_limits["requests"]) >= 60:
            # Block user for 5 minutes
            user_limits["blocked_until"] = current_time + timedelta(minutes=5)
            logger.warning(f"Rate limit exceeded for user {user_hash}, blocked for 5 minutes")
            return False
        
        # Record request
        user_limits["requests"].append(current_time)
        return True
    
    def _validate_additional_context(self, context: Dict[str, Any]) -> bool:
        """Validate additional request context"""
        
        try:
            # Check context size
            context_str = json.dumps(context)
            if len(context_str) > 10000:  # 10KB limit
                logger.warning("Additional context too large")
                return False
            
            # Validate history if present
            if "history" in context:
                history = context["history"]
                if isinstance(history, list) and len(history) > 100:
                    logger.warning("History too long")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Context validation error: {str(e)}")
            return False
    
    def _validate_prediction_structure(self, prediction: Dict[str, Any]) -> bool:
        """Validate prediction has required structure"""
        
        required_fields = ["api_call", "method", "description"]
        
        for field in required_fields:
            if field not in prediction or not prediction[field]:
                return False
        
        # Validate HTTP method
        method = prediction["method"].upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            return False
        
        return True
    
    def _validate_api_call_security(self, api_call: str) -> bool:
        """Security validation for API call paths"""
        
        # Check for path traversal in API calls
        if ".." in api_call or "/./" in api_call:
            return False
        
        # Check for suspicious endpoints
        suspicious_patterns = [
            r"(?i)(admin|system|config|debug|test)",
            r"(?i)(passwd|shadow|secret|key)",
            r"(?i)(exec|eval|cmd|shell)"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, api_call):
                logger.debug(f"Suspicious API call pattern: {pattern}")
                # Don't block, just log for monitoring
        
        return True
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate API call parameters"""
        
        try:
            # Check parameter depth and size
            params_str = json.dumps(params)
            if len(params_str) > 5000:  # 5KB limit
                return False
            
            # Check for suspicious parameter values
            for key, value in params.items():
                if isinstance(value, str):
                    # Check for injection attempts in parameter values
                    if not self._validate_security(value):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters by removing unsafe values"""
        
        sanitized = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Basic sanitization
                value = re.sub(r'[<>"\']', '', value)
                value = value[:200]  # Limit length
            elif isinstance(value, (int, float, bool)):
                pass  # Keep as is
            elif isinstance(value, (list, dict)):
                # Truncate complex structures
                value = str(value)[:100]
            
            sanitized[key] = value
        
        return sanitized
    
    def _validate_description_content(self, description: str) -> bool:
        """Validate description content"""
        return len(description) <= 500 and not self._contains_excessive_profanity(description)
    
    def _sanitize_description(self, description: str) -> str:
        """Sanitize description content"""
        # Remove HTML tags and limit length
        sanitized = re.sub(r'<[^>]+>', '', description)
        return sanitized[:200]
    
    def _contains_excessive_profanity(self, text: str) -> bool:
        """Check for excessive profanity (placeholder implementation)"""
        # PLACEHOLDER: In production, this would use a comprehensive profanity filter
        basic_profanity = ["damn", "shit", "fuck", "bitch"]  # Very basic list
        text_lower = text.lower()
        
        profanity_count = sum(1 for word in basic_profanity if word in text_lower)
        return profanity_count > 2  # Allow some casual profanity
    
    def _contains_personal_info(self, text: str) -> bool:
        """Check for personal information patterns"""
        
        # Email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return True
        
        # Phone numbers
        if re.search(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b', text):
            return True
        
        # Social Security Numbers (US format)
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            return True
        
        return False
    
    def _contains_command_injection(self, text: str) -> bool:
        """Check for command injection patterns"""
        
        command_patterns = [
            r"(?i)(;|\||\&\&|\|\|)\s*(rm\s|del\s|format\s)",
            r"(?i)(wget\s|curl\s|nc\s|netcat\s)",
            r"(?i)(>|\>>)\s*/dev/",
            r"(?i)\$\(.*\)"
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _load_blocked_patterns(self) -> List[str]:
        """Load blocked content patterns"""
        
        # Basic blocked patterns - in production this would be more comprehensive
        return [
            r"(?i)(hate\s+speech|racial\s+slur)",
            r"(?i)(bomb|terrorist|attack\s+plan)",
            r"(?i)(illegal\s+drugs|drug\s+dealing)",
            r"(?i)(child\s+abuse|exploitation)",
            # Add more patterns as needed
        ]
    
    def is_safe(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if a single prediction is safe to return to users
        
        Args:
            prediction: Dictionary containing API call prediction with fields:
                - api_call: The API endpoint/call
                - method: HTTP method (GET, POST, etc.)
                - description: Human-readable description
                - parameters: Request parameters
                
        Returns:
            True if prediction is safe, False if it should be filtered out
        """
        try:
            # Validate prediction structure first
            if not self._validate_prediction_structure(prediction):
                logger.debug("Prediction failed structure validation")
                return False
            
            # Check API call security
            api_call = prediction.get("api_call", "")
            if not self._validate_api_call_security(api_call):
                logger.debug(f"API call failed security check: {api_call}")
                return False
            
            # Check for destructive operations
            method = prediction.get("method", "").upper()
            if self._is_destructive_operation(api_call, method):
                logger.debug(f"Blocked destructive operation: {method} {api_call}")
                return False
            
            # Validate parameters
            params = prediction.get("parameters", {})
            if not self._validate_parameters(params):
                logger.debug("Parameters failed validation")
                return False
            
            # Check description content
            description = prediction.get("description", "")
            if not self._validate_description_content(description):
                logger.debug("Description content failed validation")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Safety check error for prediction: {str(e)}")
            # Fail closed - consider unsafe on error
            return False
    
    def _is_destructive_operation(self, api_call: str, method: str) -> bool:
        """Check if an API call represents a destructive operation"""
        
        # Always block DELETE operations
        if method == "DELETE":
            return True
        
        # Block dangerous endpoints regardless of method
        dangerous_patterns = [
            r"(?i)(delete|remove|purge|drop|truncate)",
            r"(?i)(admin|system|config)/.*/(delete|remove|destroy)",
            r"(?i)/(shutdown|restart|reboot|reset)",
            r"(?i)/(wipe|clear|erase)",
            r"(?i)/debug/.*/(exec|eval|run)"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, api_call):
                return True
        
        # Check for destructive operations in PUT/POST to specific endpoints
        if method in ["PUT", "POST"]:
            destructive_endpoint_patterns = [
                r"(?i)/.*/(delete|remove|purge)",
                r"(?i)/admin/.*/(destroy|wipe)",
                r"(?i)/system/.*/(reset|clear)"
            ]
            
            for pattern in destructive_endpoint_patterns:
                if re.search(pattern, api_call):
                    return True
        
        return False

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics and metrics"""
        
        blocked_rate = (
            self.blocked_requests / self.total_validations 
            if self.total_validations > 0 else 0
        )
        
        security_violation_rate = (
            self.security_violations / self.total_validations 
            if self.total_validations > 0 else 0
        )
        
        return {
            "total_validations": self.total_validations,
            "blocked_requests": self.blocked_requests,
            "security_violations": self.security_violations,
            "blocked_rate": blocked_rate,
            "security_violation_rate": security_violation_rate,
            "active_rate_limits": len([
                user for user, limits in self.rate_limits.items()
                if limits["blocked_until"] and limits["blocked_until"] > datetime.utcnow()
            ]),
            "max_prompt_length": self.max_prompt_length
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.total_validations = 0
        self.blocked_requests = 0
        self.security_violations = 0
        logger.info("Validation statistics reset")

# Convenience functions
def validate_prompt(prompt: str, user_id: Optional[str] = None) -> bool:
    """Convenience function for prompt validation"""
    validator = SafetyValidator()
    return validator.validate_input(prompt, user_id)

def filter_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function for prediction filtering"""
    validator = SafetyValidator()
    filtered, warnings = validator.validate_output(predictions)
    if warnings:
        logger.info(f"Prediction filtering warnings: {warnings}")
    return filtered

def is_prediction_safe(prediction: Dict[str, Any]) -> bool:
    """Convenience function to check if a single prediction is safe"""
    validator = SafetyValidator()
    return validator.is_safe(prediction)

def validate_api_safety(api_call: str, method: str, params: Dict[str, Any] = None) -> bool:
    """Convenience function for API safety validation with destructive pattern detection"""
    prediction = {
        "api_call": api_call,
        "method": method,
        "parameters": params or {},
        "description": f"{method} request to {api_call}"
    }
    return is_prediction_safe(prediction)