"""
Workflow Logging Module for Java Peer Review Training System.

This module provides structured logging functionality to track workflow operations,
performance metrics, and agent attributions throughout the system.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WorkflowLogger:
    """
    Structured logger for tracking workflow operations and performance metrics.
    
    This logger creates structured logs with consistent formatting and context
    to track operations across different components of the workflow.
    """
    
    def __init__(self, session_id: str, log_level: str = "INFO", 
                 log_to_file: bool = True, log_dir: str = None):
        """
        Initialize the workflow logger.
        
        Args:
            session_id: Unique session identifier for tracking
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file in addition to console
            log_dir: Directory for log files (default: exports/logs)
        """
        self.session_id = session_id
        
        # Create a unique logger for this workflow
        self.logger = logging.getLogger(f"workflow.{session_id}")
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Always log to console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Optionally log to file
        if log_to_file:
            # Create log directory if not exists
            if log_dir is None:
                # Use exports/logs directory by default
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                log_dir = os.path.join(current_dir, "exports", "logs")
                
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Create log file handler
            log_file = os.path.join(log_dir, f"workflow_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
        
        # Track performance metrics
        self.performance_metrics = {
            "start_time": time.time(),
            "operations": {},
            "timings": {},
            "agent_attribution": {}
        }
        
        # Log initialization
        self.logger.info(f"Initialized workflow logger for session {session_id}")
    
    def log_operation_start(self, operation_name: str, agent_id: str = None, 
                           details: Dict[str, Any] = None) -> str:
        """
        Log the start of an operation and start timing.
        
        Args:
            operation_name: Name of the operation
            agent_id: ID of the agent performing the operation
            details: Additional details about the operation
            
        Returns:
            Operation ID for tracking
        """
        # Generate a unique operation ID for tracking
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Log operation start
        log_entry = {
            "event": "operation_start",
            "operation_id": operation_id,
            "operation_name": operation_name,
            "agent_id": agent_id,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Track operation in performance metrics
        self.performance_metrics["operations"][operation_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "agent_id": agent_id,
            "details": details or {},
            "status": "running"
        }
        
        # Track agent attribution
        if agent_id:
            if agent_id not in self.performance_metrics["agent_attribution"]:
                self.performance_metrics["agent_attribution"][agent_id] = {
                    "operations": [],
                    "total_time": 0
                }
            self.performance_metrics["agent_attribution"][agent_id]["operations"].append(operation_id)
        
        # Log as structured data
        self.logger.info(f"Started operation: {operation_name} (ID: {operation_id})", 
                        extra={"structured": log_entry})
        
        return operation_id
    
    def log_operation_end(self, operation_id: str, status: str = "success", 
                         output_summary: Optional[str] = None, 
                         metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Log the end of an operation and record timing.
        
        Args:
            operation_id: ID of the operation from log_operation_start
            status: Status of the operation (success, failure, partial)
            output_summary: Brief summary of the operation output
            metrics: Additional metrics about the operation
            
        Returns:
            Duration of the operation in seconds
        """
        # Make sure the operation exists
        if operation_id not in self.performance_metrics["operations"]:
            self.logger.warning(f"Cannot end unknown operation ID: {operation_id}")
            return 0.0
        
        # Get operation details
        operation = self.performance_metrics["operations"][operation_id]
        operation_name = operation["name"]
        agent_id = operation["agent_id"]
        
        # Calculate timing
        end_time = time.time()
        start_time = operation["start_time"]
        duration = end_time - start_time
        
        # Update operation details
        operation["end_time"] = end_time
        operation["duration"] = duration
        operation["status"] = status
        operation["output_summary"] = output_summary
        if metrics:
            operation["metrics"] = metrics
        
        # Update timings by operation name
        if operation_name not in self.performance_metrics["timings"]:
            self.performance_metrics["timings"][operation_name] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        
        timing_stats = self.performance_metrics["timings"][operation_name]
        timing_stats["count"] += 1
        timing_stats["total_time"] += duration
        timing_stats["min_time"] = min(timing_stats["min_time"], duration)
        timing_stats["max_time"] = max(timing_stats["max_time"], duration)
        
        # Update agent attribution
        if agent_id and agent_id in self.performance_metrics["agent_attribution"]:
            self.performance_metrics["agent_attribution"][agent_id]["total_time"] += duration
        
        # Log as structured data
        log_entry = {
            "event": "operation_end",
            "operation_id": operation_id,
            "operation_name": operation_name,
            "agent_id": agent_id,
            "status": status,
            "duration": duration,
            "output_summary": output_summary,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed operation: {operation_name} (ID: {operation_id}) in {duration:.2f}s - Status: {status}", 
                        extra={"structured": log_entry})
        
        return duration
    
    def log_error(self, error_message: str, operation_id: Optional[str] = None, 
                exception: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        """
        Log an error with context.
        
        Args:
            error_message: Main error message
            operation_id: Optional ID of the operation where the error occurred
            exception: Optional exception object
            details: Additional error details
        """
        # Create structured error log
        log_entry = {
            "event": "error",
            "operation_id": operation_id,
            "message": error_message,
            "exception": str(exception) if exception else None,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add operation context if available
        if operation_id and operation_id in self.performance_metrics["operations"]:
            operation = self.performance_metrics["operations"][operation_id]
            log_entry["operation_name"] = operation["name"]
            log_entry["agent_id"] = operation["agent_id"]
            
            # Update operation status
            operation["status"] = "error"
            if exception:
                operation["error"] = str(exception)
        
        # Log the error
        error_msg = f"ERROR: {error_message}"
        if exception:
            error_msg += f" - Exception: {str(exception)}"
            
        self.logger.error(error_msg, extra={"structured": log_entry})
        
        # Return the log entry for reference
        return log_entry
        
    def log_llm_interaction(self, agent_id: str, prompt: str, response: str, 
                           operation_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Log an interaction with an LLM.
        
        Args:
            agent_id: ID of the agent performing the interaction
            prompt: The prompt sent to the LLM
            response: The response from the LLM
            operation_id: Optional ID of the parent operation
            metadata: Additional metadata about the interaction
        """
        # Create structured log entry
        log_entry = {
            "event": "llm_interaction",
            "agent_id": agent_id,
            "operation_id": operation_id,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate prompt and response statistics
        prompt_word_count = len(prompt.split())
        response_word_count = len(response.split())
        
        # Add to log entry
        log_entry["prompt_word_count"] = prompt_word_count
        log_entry["response_word_count"] = response_word_count
        
        # Update agent attribution
        if agent_id in self.performance_metrics["agent_attribution"]:
            agent_stats = self.performance_metrics["agent_attribution"][agent_id]
            if "llm_interactions" not in agent_stats:
                agent_stats["llm_interactions"] = 0
            if "total_prompt_tokens" not in agent_stats:
                agent_stats["total_prompt_tokens"] = 0
            if "total_response_tokens" not in agent_stats:
                agent_stats["total_response_tokens"] = 0
                
            agent_stats["llm_interactions"] += 1
            # Rough token estimation (words * 1.3)
            agent_stats["total_prompt_tokens"] += int(prompt_word_count * 1.3)
            agent_stats["total_response_tokens"] += int(response_word_count * 1.3)
        
        # Log the interaction
        self.logger.info(
            f"LLM interaction by {agent_id}: {prompt_word_count} words in, {response_word_count} words out",
            extra={"structured": log_entry}
        )
        
        # Also save the full prompt and response to a file if verbose logging is enabled
        if hasattr(self, 'verbose') and self.verbose and self.log_file:
            log_dir = os.path.dirname(self.log_file)
            llm_log_file = os.path.join(
                log_dir, 
                f"llm_{self.session_id}_{int(time.time() * 1000)}.txt"
            )
            
            with open(llm_log_file, 'w', encoding='utf-8') as f:
                f.write(f"AGENT: {agent_id}\n")
                f.write(f"OPERATION: {operation_id}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n\n")
                f.write("PROMPT:\n")
                f.write("="*80 + "\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("RESPONSE:\n")
                f.write("="*80 + "\n")
                f.write(response)
        
        return log_entry
    
    def log_state_transition(self, node_name: str, state_before: Any, state_after: Any, 
                            operation_id: Optional[str] = None):
        """
        Log a state transition in the workflow.
        
        Args:
            node_name: Name of the node causing the state transition
            state_before: State before the transition
            state_after: State after the transition
            operation_id: Optional ID of the parent operation
        """
        # Create a simplified state representation to avoid huge log entries
        def simplify_state(state):
            if hasattr(state, "__dict__"):
                return {
                    k: "..." if k in ["code_snippet", "review_history"] 
                    else simplify_state(v) for k, v in state.__dict__.items()
                }
            elif isinstance(state, dict):
                return {k: "..." if k in ["code", "clean_code"] else v for k, v in state.items()}
            else:
                return state
        
        # Simplify states
        simple_before = simplify_state(state_before)
        simple_after = simplify_state(state_after)
        
        # Create diff of key state changes
        diff = {}
        if hasattr(state_after, "__dict__"):
            for k, v in state_after.__dict__.items():
                # Check if attribute exists in before state
                if hasattr(state_before, k):
                    before_value = getattr(state_before, k)
                    if before_value != v:
                        diff[k] = {
                            "before": str(before_value)[:100] + "..." if isinstance(before_value, str) and len(str(before_value)) > 100 else before_value,
                            "after": str(v)[:100] + "..." if isinstance(v, str) and len(str(v)) > 100 else v
                        }
                else:
                    diff[k] = {
                        "before": None,
                        "after": str(v)[:100] + "..." if isinstance(v, str) and len(str(v)) > 100 else v
                    }
        
        # Create structured log entry
        log_entry = {
            "event": "state_transition",
            "node_name": node_name,
            "operation_id": operation_id,
            "state_changes": diff,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the transition
        self.logger.info(
            f"State transition in node '{node_name}' - Changed fields: {', '.join(diff.keys())}",
            extra={"structured": log_entry}
        )
        
        return log_entry
    
    def log_workflow_node(self, node_name: str, entry_or_exit: str = "entry", 
                        details: Optional[Dict[str, Any]] = None):
        """
        Log entry or exit of a workflow node.
        
        Args:
            node_name: Name of the workflow node
            entry_or_exit: Either "entry" or "exit"
            details: Additional details about the node execution
        """
        # Create structured log entry
        log_entry = {
            "event": f"node_{entry_or_exit}",
            "node_name": node_name,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the node entry/exit
        self.logger.info(
            f"Node {entry_or_exit}: {node_name}",
            extra={"structured": log_entry}
        )
        
        return log_entry
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics for this workflow.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate total workflow time
        total_time = time.time() - self.performance_metrics["start_time"]
        
        # Calculate average times for operations
        operation_stats = {}
        for op_name, timing in self.performance_metrics["timings"].items():
            avg_time = timing["total_time"] / timing["count"] if timing["count"] > 0 else 0
            operation_stats[op_name] = {
                "count": timing["count"],
                "avg_time": avg_time,
                "min_time": timing["min_time"] if timing["min_time"] != float('inf') else 0,
                "max_time": timing["max_time"],
                "total_time": timing["total_time"]
            }
        
        # Create summary
        summary = {
            "session_id": self.session_id,
            "total_time": total_time,
            "operation_stats": operation_stats,
            "agent_stats": self.performance_metrics["agent_attribution"],
            "total_operations": len(self.performance_metrics["operations"]),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def export_logs(self, export_path: Optional[str] = None) -> str:
        """
        Export all logs and performance metrics to a JSON file.
        
        Args:
            export_path: Optional path for the export file
            
        Returns:
            Path to the exported file
        """
        # Create default export path if not provided
        if export_path is None:
            if self.log_file:
                log_dir = os.path.dirname(self.log_file)
                export_path = os.path.join(
                    log_dir, 
                    f"workflow_metrics_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            else:
                # Use current directory if no log file
                export_path = f"workflow_metrics_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get performance summary
        summary = self.get_performance_summary()
        
        # Add all operations
        summary["operations"] = self.performance_metrics["operations"]
        
        # Export to file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Exported workflow metrics to {export_path}")
        
        return export_path
    
    def set_verbose(self, verbose: bool = True):
        """
        Set verbose logging mode.
        
        Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        self.logger.info(f"Set verbose logging mode to {verbose}")
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up when exiting context manager.
        Export logs automatically.
        """
        # Log any exception
        if exc_type is not None:
            self.log_error(
                error_message=f"Workflow error in context manager: {exc_type.__name__}",
                exception=exc_val
            )
        
        # Calculate final metrics
        summary = self.get_performance_summary()
        
        # Log completion
        self.logger.info(
            f"Workflow completed in {summary['total_time']:.2f}s with {summary['total_operations']} operations",
            extra={"structured": {"event": "workflow_complete", "summary": summary}}
        )
        
        # Export logs
        self.export_logs()
        
        # Don't suppress exceptions
        return False