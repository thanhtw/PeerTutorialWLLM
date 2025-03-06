"""
Service package for Java Peer Review Training System.

This package contains service components that coordinate the
workflow between the UI, domain objects, and LLM manager.
"""

from service.agent_service import AgentService

__all__ = [
    'AgentService'
]