# Pragmatic Error Handling Improvements for LLMProc

## Executive Summary

Without introducing new error classes, we can still significantly improve LLMProc's error handling by standardizing existing patterns, improving logging, and ensuring consistent error propagation across the codebase. This document outlines practical improvements that can be implemented incrementally while working within the current architecture.

## Practical Improvement Areas

### 1. Standardize Error Messages

Implement consistent error message formatting across the codebase:

```python
# Before (inconsistent formats across the codebase):
raise ValueError("Model not found")
raise RuntimeError(f"Failed to call provider API: {str(e)}")

# After (consistent format with context):
raise ValueError(f"Model '{model_name}' not found for provider '{provider}'")
raise RuntimeError(f"Failed to call {provider} API: {str(e)}")
```

### 2. Improve Logging Consistency

Standardize logging levels and formats:

```python
# Before (inconsistent logging):
logger.error(f"API call failed: {e}")  # Some components
logger.warning(f"Rate limit hit: {e}") # Other components use different levels
print(f"Error: {e}")                   # Some use print instead of logging

# After (consistent logging):
logger = logging.getLogger(__name__)  # Module-level logger

# Error for issues requiring immediate attention
logger.error(f"{provider}: API authentication failed: {e}")

# Warning for important but non-critical issues
logger.warning(f"{provider}: Rate limit reached ({e}), backing off")

# Info for normal but significant events
logger.info(f"{provider}: Request completed in {duration}ms")

# Debug for detailed diagnostic information
logger.debug(f"{provider}: API request: {api_params}")
```

### 3. Centralize Common Error Handling Logic

Create utility functions for common error handling patterns without introducing new classes:

```python
def log_api_error(logger, provider, error, context=None):
    """Log API errors consistently."""
    context_str = f" | Context: {context}" if context else ""
    logger.error(f"{provider} API error: {str(error)}{context_str}")

    # Add standard diagnostic information
    if hasattr(error, "status_code"):
        logger.debug(f"Status code: {error.status_code}")

    # Log rate limits differently
    if "rate limit" in str(error).lower() or getattr(error, "status_code", 0) == 429:
        logger.warning(f"{provider} rate limit reached, consider implementing backoff")


def format_error_message(provider, operation, error, details=None):
    """Format error messages consistently."""
    msg = f"{provider}: {operation} failed: {str(error)}"
    if details:
        msg += f" ({details})"
    return msg
```

### 4. Improve Error Propagation

Ensure errors propagate with sufficient context:

```python
# Before
try:
    response = await client.messages.create(**params)
except Exception as e:
    raise RuntimeError(f"API call failed: {str(e)}")  # Loses context

# After
try:
    response = await client.messages.create(**params)
except Exception as e:
    logger.error(f"{provider}: API call failed: {str(e)}")
    # Re-raise with context but preserves original exception as cause
    raise RuntimeError(f"{provider} API call failed: {str(e)}") from e
```

### 5. Add Request IDs for Correlation

Add request IDs to help correlate errors across components:

```python
def generate_request_id():
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]  # Short UUID

# In the run method
request_id = generate_request_id()
logger.info(f"Starting request {request_id} for model {process.model_name}")

try:
    # Make the API call with request_id in the log context
    logger.debug(f"[{request_id}] Calling {provider} API")
    response = await client.messages.create(**params)
except Exception as e:
    logger.error(f"[{request_id}] {provider} API call failed: {str(e)}")
    raise
```

### 6. Add Provider-Specific Error Handling

Improve handling of known provider-specific errors without new classes:

```python
def handle_anthropic_error(error, logger):
    """Handle Anthropic-specific errors."""
    error_msg = str(error).lower()

    if "api key" in error_msg or getattr(error, "status_code", 0) == 401:
        logger.error("Anthropic API key validation failed")
        # Forward with specific message
        return f"Authentication error: Please check your Anthropic API key"

    elif "rate limit" in error_msg or getattr(error, "status_code", 0) == 429:
        logger.warning("Anthropic rate limit reached")
        return f"Rate limit reached: Please reduce request frequency"

    # Default handler
    return f"Anthropic API error: {str(error)}"


# In anthropic_process_executor.py
try:
    response = await client.messages.create(**params)
except Exception as e:
    error_msg = handle_anthropic_error(e, logger)
    raise RuntimeError(error_msg) from e
```

### 7. Improve RunResult Error Tracking

Enhance the existing RunResult class to better track errors:

```python
# In results.py
@dataclass
class RunResult:
    # Existing fields...
    error: Optional[Exception] = None
    error_context: Optional[dict] = None

    def add_error(self, error: Exception, context: dict = None):
        """Add error information to the result."""
        self.error = error
        self.error_context = context or {}
        return self

    def has_error(self) -> bool:
        """Check if the run resulted in an error."""
        return self.error is not None
```

### 8. Add Retry For Common Errors

Implement simple retry logic for transient errors without complex machinery:

```python
async def call_with_retry(func, max_retries=3, initial_backoff=1):
    """Call a function with retry logic for common transient errors."""
    retries = 0
    backoff = initial_backoff

    while True:
        try:
            return await func()
        except Exception as e:
            retries += 1

            # Check if we've exhausted retries
            if retries >= max_retries:
                logger.warning(f"Max retries ({max_retries}) exceeded")
                raise  # Re-raise the last exception

            # Check if this is a retryable error
            if "rate limit" in str(e).lower() or getattr(e, "status_code", 0) == 429:
                wait_time = backoff * (2 ** (retries - 1))  # Exponential backoff
                logger.info(f"Rate limit hit, retrying in {wait_time}s ({retries}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue

            # Non-retryable error
            raise
```

## Implementation Plan

### Phase 1: Core Improvements (Priority: High)

1. **Standardize logging** across all providers
   - Use consistent logging levels
   - Ensure all components use the logging module, not print
   - Add request ID for correlation

2. **Improve error context** in all raise statements
   - Include provider name
   - Include operation description
   - Preserve original exception with `raise X from e`

### Phase 2: Utility Functions (Priority: Medium)

1. **Create common error handling utilities**
   - `log_api_error` function
   - `format_error_message` function
   - Provider-specific error handling functions

2. **Enhance existing RunResult**
   - Add error tracking
   - Add error context

### Phase 3: Resilience (Priority: Medium)

1. **Add simple retry mechanisms**
   - Implement for rate limiting
   - Add exponential backoff

2. **Improve error recovery**
   - Add graceful degradation where possible
   - Add fallback mechanisms

## Key Benefits

This approach offers several benefits:

1. **Incremental improvement**: Can be implemented gradually without major refactoring
2. **Consistency**: Standardized error messages and logging
3. **Correlation**: Request IDs help track errors across components
4. **Context preservation**: Better error propagation retains diagnostic information
5. **Resilience**: Simple retry mechanisms improve reliability
6. **No new classes**: Works within existing architecture

## Example Implementation in anthropic_process_executor.py

Here's how these improvements might look in the Anthropic process executor:

```python
async def run(self, process, user_prompt, max_iterations=10, callbacks=None, run_result=None):
    """Execute a conversation with the Anthropic API."""
    callbacks = callbacks or {}
    run_result = run_result or RunResult()
    request_id = generate_request_id()

    logger.info(f"[{request_id}] Starting Anthropic request for model {process.model_name}")

    # Add user prompt to state
    if not is_tool_continuation:
        process.state.append({"role": "user", "content": user_prompt})

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        logger.debug(f"[{request_id}] Making API call {iterations}/{max_iterations}")

        try:
            # Make the API call
            response = await process.client.messages.create(
                model=process.model_name,
                system=process.enriched_system_prompt,
                messages=process.state,
                tools=process.tools,
                **process.api_params,
            )

            # Track in run result
            run_result.add_api_call({
                "model": process.model_name,
                "request_id": request_id,
                "usage": getattr(response, "usage", {}),
                "stop_reason": getattr(response, "stop_reason", None),
            })

            # Process response...

        except Exception as e:
            # Log error with context
            error_context = {
                "request_id": request_id,
                "model": process.model_name,
                "iteration": iterations,
                "provider": "anthropic",
            }
            log_api_error(logger, "anthropic", e, error_context)

            # Add error to run result
            run_result.add_error(e, error_context)
            process.run_stop_reason = "error"

            # Re-raise with context
            raise RuntimeError(
                f"Anthropic API error on request {request_id}: {str(e)}"
            ) from e

    return run_result.complete()
```

## Conclusion

By focusing on standardization, better context preservation, and consistent logging, we can significantly improve error handling in LLMProc without introducing new error classes. These pragmatic improvements can be implemented incrementally while providing immediate benefits in terms of diagnostics, reliability, and developer experience.
