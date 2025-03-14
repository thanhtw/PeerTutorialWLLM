"""
Feedback Display UI module for Java Peer Review Training System.

This module provides the FeedbackDisplayUI class for displaying feedback on student reviews.
"""

import streamlit as st
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackDisplayUI:
    """
    UI Component for displaying feedback on student reviews.
    
    This class handles displaying analysis results, review history,
    and feedback on student reviews.
    """
    
    def render_results(self, 
                      comparison_report: str = None,
                      review_summary: str = None,
                      review_analysis: Dict[str, Any] = None,
                      review_history: List[Dict[str, Any]] = None,
                      on_reset_callback: Callable[[], None] = None) -> None:
        """
        Render the analysis results and feedback with improved review visibility.
        
        Args:
            comparison_report: Comparison report text
            review_summary: Review summary text
            review_analysis: Analysis of student review
            review_history: History of review iterations
            on_reset_callback: Callback function when reset button is clicked
        """
        if not comparison_report and not review_summary and not review_analysis:
            st.info("No analysis results available. Please submit your review in the 'Submit Review' tab first.")
            return
        
        # First show performance summary metrics at the top
        if review_history and len(review_history) > 0 and review_analysis:
            self._render_performance_summary(review_analysis, review_history)
        
        # Display the comparison report
        if comparison_report:
            st.subheader("Educational Feedback:")
            st.markdown(
                f'<div class="comparison-report">{comparison_report}</div>',
                unsafe_allow_html=True
            )
        
        # Always show review history for better visibility
        if review_history and len(review_history) > 0:
            st.subheader("Your Review:")
            
            # First show the most recent review prominently
            if review_history:
                latest_review = review_history[-1]
                review_analysis = latest_review.get("review_analysis", {})
                iteration = latest_review.get("iteration_number", 0)
                
                st.markdown(f"#### Your Final Review (Attempt {iteration})")
                
                # Format the review text with syntax highlighting
                st.markdown("```text\n" + latest_review.get("student_review", "") + "\n```")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Issues Found", 
                        f"{review_analysis.get('identified_count', 0)} of {review_analysis.get('total_problems', 0)}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Accuracy", 
                        f"{review_analysis.get('accuracy_percentage', 0):.1f}%",
                        delta=None
                    )
                with col3:
                    false_positives = len(review_analysis.get('false_positives', []))
                    st.metric(
                        "False Positives", 
                        false_positives,
                        delta=None
                    )
            
            # Show earlier reviews in an expander if there are multiple
            if len(review_history) > 1:
                with st.expander("Review History", expanded=False):
                    tabs = st.tabs([f"Attempt {rev.get('iteration_number', i+1)}" for i, rev in enumerate(review_history)])
                    
                    for i, (tab, review) in enumerate(zip(tabs, review_history)):
                        with tab:
                            review_analysis = review.get("review_analysis", {})
                            st.markdown("```text\n" + review.get("student_review", "") + "\n```")
                            
                            st.write(f"**Found:** {review_analysis.get('identified_count', 0)} of "
                                    f"{review_analysis.get('total_problems', 0)} issues "
                                    f"({review_analysis.get('accuracy_percentage', 0):.1f}% accuracy)")
        
        # Display analysis details in an expander
        if review_summary or review_analysis:
            with st.expander("Detailed Analysis", expanded=True):
                tabs = st.tabs(["Identified Issues", "Missed Issues", "False Positives", "Summary"])
                
                with tabs[0]:  # Identified Issues
                    self._render_identified_issues(review_analysis)
                
                with tabs[1]:  # Missed Issues
                    self._render_missed_issues(review_analysis)
                
                with tabs[2]:  # False Positives
                    self._render_false_positives(review_analysis)
                
                with tabs[3]:  # Summary
                    if review_summary:
                        st.markdown(review_summary)
        
        # Download button for feedback report
        #if review_summary:
            # Create a dynamic key based on the content
            #feedback_key = f"download_feedback_{hash(review_summary)%10000}"
            
            # if st.download_button(
            #     label="Download Feedback Report", 
            #     data=review_summary,
            #     file_name="java_review_feedback.md",
            #     mime="text/markdown",
            #     key='download2'
            # ):
            #     st.success("Feedback report downloaded successfully!")
        
        # Start over button
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Start New Review", type="primary", use_container_width=True):
                # Call the reset callback if provided
                if on_reset_callback:
                    on_reset_callback()
                else:
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        # Keep error selection mode and categories
                        if key not in ["error_selection_mode", "selected_error_categories"]:
                            del st.session_state[key]
                    
                    # Rerun the app
                    st.rerun()
    
    def _render_performance_summary(self, review_analysis: Dict[str, Any], review_history: List[Dict[str, Any]]):
        """Render performance summary metrics and charts"""
        st.subheader("Review Performance Summary")
        
        # Create performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = review_analysis.get("accuracy_percentage", 0)
            st.metric(
                "Overall Accuracy", 
                f"{accuracy:.1f}%",
                delta=None
            )
            
        with col2:
            found = review_analysis.get("identified_count", 0)
            total = review_analysis.get("total_problems", 0)
            st.metric(
                "Issues Identified", 
                f"{found}/{total}",
                delta=None
            )
            
        with col3:
            false_positives = len(review_analysis.get("false_positives", []))
            st.metric(
                "False Positives", 
                f"{false_positives}",
                delta=None
            )
            
        # Create a progress chart if multiple iterations
        if len(review_history) > 1:
            # Extract data for chart
            iterations = []
            identified_counts = []
            accuracy_percentages = []
            
            for review in review_history:
                analysis = review.get("review_analysis", {})
                iterations.append(review.get("iteration_number", 0))
                identified_counts.append(analysis.get("identified_count", 0))
                accuracy_percentages.append(analysis.get("accuracy_percentage", 0))
                
            # Create a DataFrame for the chart
            chart_data = pd.DataFrame({
                "Iteration": iterations,
                "Issues Found": identified_counts,
                "Accuracy (%)": accuracy_percentages
            })
            
            # Display the chart with two y-axes
            st.subheader("Progress Across Iterations")
            
            # Using matplotlib for more control
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Issues Found', color=color)
            ax1.plot(chart_data["Iteration"], chart_data["Issues Found"], marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            ax2 = ax1.twinx()  # Create a second y-axis
            color = 'tab:red'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(chart_data["Iteration"], chart_data["Accuracy (%)"], marker='s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            st.pyplot(fig)
    
    def _render_identified_issues(self, review_analysis: Dict[str, Any]):
        """Render identified issues section"""
        identified_problems = review_analysis.get("identified_problems", [])
        
        if not identified_problems:
            st.info("You didn't identify any issues correctly.")
            return
            
        st.subheader(f"Correctly Identified Issues ({len(identified_problems)})")
        
        for i, issue in enumerate(identified_problems, 1):
            st.markdown(
                f"""
                <div style="border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>✓ {i}. {issue}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    def _render_missed_issues(self, review_analysis: Dict[str, Any]):
        """Render missed issues section"""
        missed_problems = review_analysis.get("missed_problems", [])
        
        if not missed_problems:
            st.success("Great job! You identified all the issues.")
            return
            
        st.subheader(f"Issues You Missed ({len(missed_problems)})")
        
        for i, issue in enumerate(missed_problems, 1):
            st.markdown(
                f"""
                <div style="border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>✗ {i}. {issue}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    def _render_false_positives(self, review_analysis: Dict[str, Any]):
        """Render false positives section"""
        false_positives = review_analysis.get("false_positives", [])
        
        if not false_positives:
            st.success("You didn't report any false positives! Good job distinguishing real issues.")
            return
            
        st.subheader(f"False Positives ({len(false_positives)})")
        
        for i, issue in enumerate(false_positives, 1):
            st.markdown(
                f"""
                <div style="border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>⚠ {i}. {issue}</strong>
                <p>This wasn't actually an issue in the code.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )