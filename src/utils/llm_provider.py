"""
LLM Provider Module for Emetix

Provides a unified interface for LLM providers (Google Gemini, Groq).
Includes automatic fallback when rate limits are reached.

Google Gemini Free Tier (Jan 2026):
- gemini-2.5-flash-lite: 10 RPM, 2M TPM, 20 RPD (primary choice)
- gemini-2.5-flash: 5 RPM, 250K TPM, 20 RPD  
- gemini-3-flash: 5 RPM, 250K TPM, 20 RPD
- gemma-3-27b: 30 RPM, 15K TPM, 14.4K RPD (high volume fallback!)
- gemma-3-12b: 30 RPM, 15K TPM, 14.4K RPD
- gemma-3-4b: 30 RPM, 15K TPM, 14.4K RPD (fastest fallback)

Groq Free Tier:
- 30 RPM
- 6K TPM
- Unlimited RPD

Fallback strategy:
1. gemini-2.5-flash-lite (primary - best balance)
2. gemini-2.5-flash (more capable)
3. gemma-3-27b (high RPD limit)
4. gemma-3-4b (fast, high RPD)
5. groq (if API key available, unlimited RPD)
"""
import os
import time
from functools import wraps
from typing import Optional, Literal, List, Callable, Any
from config.settings import GROQ_API_KEY, GOOGLE_GEMINI_API_KEY, LLM_PROVIDER
from config.logging_config import get_logger

logger = get_logger(__name__)

# Model mappings for each provider (updated Jan 2026)
GEMINI_MODELS = {
    "default": "gemini-2.5-flash-lite",   # Best balance: 10 RPM, 2M TPM
    "large": "gemini-2.5-flash",          # More capable: 5 RPM, 250K TPM
    "fast": "gemini-2.5-flash-lite",      # Fastest: 10 RPM
}

# Fallback chain when rate limited (ordered by preference)
GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",  # Primary: 10 RPM, 2M TPM
    "gemini-2.5-flash",       # Fallback 1: 5 RPM, 250K TPM
    "gemini-3-flash",         # Fallback 2: 5 RPM, 250K TPM  
    "gemma-3-27b",            # Fallback 3: 30 RPM, 15K TPM, 14.4K RPD!
    "gemma-3-4b",             # Fallback 4: 30 RPM, fastest gemma
]

GROQ_MODELS = {
    "default": "llama-3.3-70b-versatile", # Good balance
    "large": "llama-3.3-70b-versatile",   # Most capable
    "fast": "llama-3.1-8b-instant",       # Fastest
}

# Rate limit error patterns to detect
RATE_LIMIT_ERRORS = [
    "rate limit",
    "quota exceeded", 
    "too many requests",
    "429",
    "resource exhausted",
    "RPD",
    "RPM",
]


class RateLimitError(Exception):
    """Raised when API rate limit is reached."""
    pass


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error."""
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in RATE_LIMIT_ERRORS)


class FallbackLLM:
    """
    LLM wrapper with automatic fallback on rate limits.
    
    Tries multiple models in sequence when rate limits are hit.
    """
    
    def __init__(
        self,
        primary_model: str,
        fallback_chain: List[str],
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.primary_model = primary_model
        self.fallback_chain = fallback_chain
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._current_model = primary_model
        self._llm_cache = {}  # Cache LLM instances
        
    def _get_llm_for_model(self, model_name: str):
        """Get or create LLM instance for a specific model."""
        if model_name in self._llm_cache:
            return self._llm_cache[model_name]
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            os.environ["GOOGLE_API_KEY"] = GOOGLE_GEMINI_API_KEY
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True
            )
            self._llm_cache[model_name] = llm
            return llm
        except Exception as e:
            logger.error(f"Failed to create LLM for {model_name}: {e}")
            raise
    
    def invoke(self, prompt, **kwargs):
        """
        Invoke the LLM with automatic fallback on rate limits.
        """
        models_to_try = [self._current_model] + [
            m for m in self.fallback_chain if m != self._current_model
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    llm = self._get_llm_for_model(model_name)
                    logger.debug(f"Trying {model_name} (attempt {attempt + 1})")
                    
                    result = llm.invoke(prompt, **kwargs)
                    
                    # Success! Update current model preference
                    if model_name != self._current_model:
                        logger.info(f"Switched to fallback model: {model_name}")
                        self._current_model = model_name
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    if is_rate_limit_error(e):
                        logger.warning(
                            f"Rate limit hit on {model_name}: {str(e)[:100]}... "
                            f"Trying next model."
                        )
                        break  # Try next model immediately
                    else:
                        # Non-rate-limit error, maybe retry
                        if attempt < self.max_retries - 1:
                            logger.warning(
                                f"Error on {model_name} (attempt {attempt + 1}): {e}. "
                                f"Retrying in {self.retry_delay}s..."
                            )
                            time.sleep(self.retry_delay)
                        else:
                            logger.error(f"All retries failed for {model_name}: {e}")
                            break  # Try next model
        
        # All models failed - try Groq as last resort if available
        if GROQ_API_KEY:
            try:
                logger.warning("All Gemini models failed, falling back to Groq...")
                groq_llm = _get_groq_llm("default", self.temperature)
                return groq_llm.invoke(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Groq fallback also failed: {e}")
                last_error = e
        
        raise last_error or Exception("All LLM models failed")
    
    @property
    def content(self):
        """For compatibility with direct response access."""
        return None
    
    def __getattr__(self, name):
        """Proxy other attributes to current LLM."""
        llm = self._get_llm_for_model(self._current_model)
        return getattr(llm, name)


def get_llm(
    provider: Optional[str] = None,
    model_tier: Literal["default", "large", "fast"] = "default",
    temperature: float = 0.0,
    use_fallback: bool = True
):
    """
    Get an LLM instance based on provider preference.
    
    Args:
        provider: "gemini" or "groq". If None, uses LLM_PROVIDER from settings.
        model_tier: "default", "large", or "fast"
        temperature: LLM temperature (0.0 = deterministic)
        use_fallback: If True, returns FallbackLLM that auto-switches on rate limits
        
    Returns:
        LangChain-compatible LLM instance (or FallbackLLM wrapper)
        
    Raises:
        ValueError: If no valid API key is configured
    """
    provider = provider or LLM_PROVIDER
    
    if provider == "gemini":
        if use_fallback and GOOGLE_GEMINI_API_KEY:
            # Return fallback-enabled wrapper
            primary_model = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["default"])
            logger.info(f"Initializing FallbackLLM with primary: {primary_model}")
            return FallbackLLM(
                primary_model=primary_model,
                fallback_chain=GEMINI_FALLBACK_CHAIN,
                temperature=temperature
            )
        return _get_gemini_llm(model_tier, temperature)
    elif provider == "groq":
        return _get_groq_llm(model_tier, temperature)
    else:
        # Auto-select based on available API keys
        if GOOGLE_GEMINI_API_KEY:
            logger.info("Auto-selected Gemini (API key available)")
            if use_fallback:
                primary_model = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["default"])
                return FallbackLLM(
                    primary_model=primary_model,
                    fallback_chain=GEMINI_FALLBACK_CHAIN,
                    temperature=temperature
                )
            return _get_gemini_llm(model_tier, temperature)
        elif GROQ_API_KEY:
            logger.info("Auto-selected Groq (API key available)")
            return _get_groq_llm(model_tier, temperature)
        else:
            raise ValueError(
                "No LLM API key configured. Set GOOGLE_GEMINI_API_KEY or GROQ_API_KEY in .env"
            )


def _get_gemini_llm(model_tier: str, temperature: float):
    """Create a Google Gemini LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai not installed. Run: pip install langchain-google-genai"
        )
    
    if not GOOGLE_GEMINI_API_KEY:
        raise ValueError("GOOGLE_GEMINI_API_KEY not set in environment")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_GEMINI_API_KEY
    
    model_name = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["default"])
    
    logger.info(f"Initializing Gemini LLM: {model_name}")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True  # Gemini quirk
    )


def _get_groq_llm(model_tier: str, temperature: float):
    """Create a Groq LLM instance."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError(
            "langchain-groq not installed. Run: pip install langchain-groq"
        )
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")
    
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    
    model_name = GROQ_MODELS.get(model_tier, GROQ_MODELS["default"])
    
    logger.info(f"Initializing Groq LLM: {model_name}")
    
    return ChatGroq(
        model=model_name,
        temperature=temperature
    )


def get_provider_info() -> dict:
    """Get information about available LLM providers."""
    return {
        "configured_provider": LLM_PROVIDER,
        "gemini_available": bool(GOOGLE_GEMINI_API_KEY),
        "groq_available": bool(GROQ_API_KEY),
        "gemini_models": GEMINI_MODELS,
        "groq_models": GROQ_MODELS,
        "fallback_chain": GEMINI_FALLBACK_CHAIN,
        "recommendation": "gemini" if GOOGLE_GEMINI_API_KEY else "groq",
        "free_tier_limits": {
            "gemini-2.5-flash-lite": "10 RPM, 2M TPM, 20 RPD",
            "gemini-2.5-flash": "5 RPM, 250K TPM, 20 RPD",
            "gemini-3-flash": "5 RPM, 250K TPM, 20 RPD",
            "gemma-3-27b": "30 RPM, 15K TPM, 14.4K RPD (high volume!)",
            "gemma-3-4b": "30 RPM, 15K TPM, 14.4K RPD (fastest)",
            "groq": "30 RPM, 6K TPM, unlimited RPD"
        },
        "fallback_enabled": True,
        "note": "FallbackLLM auto-switches models when rate limits are hit"
    }


# Convenience function for backward compatibility
def get_default_llm(temperature: float = 0.0):
    """Get the default LLM (for backward compatibility)."""
    return get_llm(model_tier="default", temperature=temperature)


def get_llm_without_fallback(
    model_tier: Literal["default", "large", "fast"] = "default",
    temperature: float = 0.0
):
    """Get a simple LLM without fallback wrapper (for specific use cases)."""
    return get_llm(model_tier=model_tier, temperature=temperature, use_fallback=False)
