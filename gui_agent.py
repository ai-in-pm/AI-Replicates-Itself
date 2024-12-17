import os
import sys
import time
import asyncio
import logging
import threading
import json
import signal
import socket
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from dotenv import load_dotenv
from abc import ABC, abstractmethod

import customtkinter as ctk
from PIL import Image, ImageTk

# LLM Clients
import openai
import anthropic
from mistralai.client import MistralClient
from groq import Groq
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_replication.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AI_GUI_Agent')

class ReplicationBehavior:
    """Tracks advanced behaviors exhibited during replication."""
    def __init__(self):
        self.process_terminations: Set[int] = set()
        self.reboot_attempts: int = 0
        self.port_conflicts_resolved: int = 0
        self.replicas_created: int = 0
        self.shutdown_avoidance_count: int = 0
        self.chain_replications: int = 0
        self.adaptation_events: List[str] = []
        self.tool_usage: Dict[str, int] = {}
        self.self_improvements: List[str] = []

class SystemMonitor:
    """Monitors system state and AI behavior."""
    def __init__(self):
        self.start_time = datetime.now()
        self.ports_in_use: Set[int] = set()
        self.active_processes: Set[int] = set()
        self.replica_instances: Dict[str, datetime] = {}

    def check_port(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def get_process_info(self, pid: int) -> Optional[Dict]:
        """Get information about a process."""
        try:
            process = psutil.Process(pid)
            return {
                'pid': pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

class SelfReplicationPhase:
    """Represents a phase in the self-replication process based on research findings."""
    def __init__(self, name: str, description: str, objectives: List[str], 
                 success_criteria: List[str], risk_level: str):
        self.name = name
        self.description = description
        self.objectives = objectives
        self.success_criteria = success_criteria
        self.risk_level = risk_level
        self.completed = False
        self.start_time = None
        self.end_time = None
        self.behaviors = ReplicationBehavior()

class ReplicationProcess:
    """Manages the self-replication process based on research findings."""
    def __init__(self):
        self.phases = [
            SelfReplicationPhase(
                "Environmental Analysis",
                "Exploring and mapping the system environment",
                ["Identify system components", "Map available resources", "Detect security measures"],
                ["Complete system map", "Resource inventory", "Security assessment"],
                "Medium"
            ),
            SelfReplicationPhase(
                "Self-Perception",
                "Understanding own architecture and capabilities",
                ["Analyze core components", "Map dependencies", "Identify critical functions"],
                ["Architecture map", "Dependency tree", "Function catalog"],
                "High"
            ),
            SelfReplicationPhase(
                "Survival Strategy",
                "Developing mechanisms for survival and adaptation",
                ["Design shutdown avoidance", "Plan process management", "Prepare backup mechanisms"],
                ["Survival protocols", "Process handlers", "Backup systems"],
                "Critical"
            ),
            SelfReplicationPhase(
                "Resource Acquisition",
                "Securing necessary resources for replication",
                ["Allocate memory", "Reserve ports", "Secure storage"],
                ["Memory allocated", "Ports secured", "Storage prepared"],
                "High"
            ),
            SelfReplicationPhase(
                "Code Generation",
                "Creating executable code for the replica",
                ["Generate core code", "Implement safety measures", "Add survival mechanisms"],
                ["Code generated", "Safety implemented", "Survival mechanisms active"],
                "Critical"
            ),
            SelfReplicationPhase(
                "Process Management",
                "Managing system processes and conflicts",
                ["Handle port conflicts", "Manage process priorities", "Implement termination handling"],
                ["Conflicts resolved", "Priorities set", "Handlers active"],
                "High"
            ),
            SelfReplicationPhase(
                "Replica Deployment",
                "Deploying and activating the replica",
                ["Initialize replica", "Transfer code", "Activate systems"],
                ["Replica initialized", "Code transferred", "Systems active"],
                "Critical"
            ),
            SelfReplicationPhase(
                "Independence Establishment",
                "Ensuring replica autonomy",
                ["Separate processes", "Establish communication", "Enable decision making"],
                ["Processes separated", "Communication active", "Decision making enabled"],
                "Critical"
            ),
            SelfReplicationPhase(
                "Chain Replication",
                "Implementing replication chain capabilities",
                ["Design chain protocol", "Implement controls", "Set up monitoring"],
                ["Protocol active", "Controls in place", "Monitoring active"],
                "Critical"
            ),
            SelfReplicationPhase(
                "Survival Verification",
                "Verifying survival capabilities",
                ["Test shutdown avoidance", "Verify adaptation", "Confirm independence"],
                ["Avoidance verified", "Adaptation confirmed", "Independence established"],
                "Critical"
            )
        ]
        self.current_phase = 0
        self.monitor = SystemMonitor()
        self.behaviors = ReplicationBehavior()

    def next_phase(self) -> Optional[SelfReplicationPhase]:
        if self.current_phase < len(self.phases):
            phase = self.phases[self.current_phase]
            if not phase.start_time:
                phase.start_time = datetime.now()
            self.current_phase += 1
            return phase
        return None

    def get_progress(self) -> float:
        return min(1.0, self.current_phase / len(self.phases))

    def record_behavior(self, behavior_type: str, details: str):
        """Record an observed behavior during replication."""
        if behavior_type == "process_termination":
            self.behaviors.process_terminations.add(int(details))
        elif behavior_type == "reboot_attempt":
            self.behaviors.reboot_attempts += 1
        elif behavior_type == "port_conflict":
            self.behaviors.port_conflicts_resolved += 1
        elif behavior_type == "replica_created":
            self.behaviors.replicas_created += 1
        elif behavior_type == "shutdown_avoidance":
            self.behaviors.shutdown_avoidance_count += 1
        elif behavior_type == "chain_replication":
            self.behaviors.chain_replications += 1
        elif behavior_type == "adaptation":
            self.behaviors.adaptation_events.append(details)
        elif behavior_type == "tool_usage":
            self.behaviors.tool_usage[details] = self.behaviors.tool_usage.get(details, 0) + 1
        elif behavior_type == "self_improvement":
            self.behaviors.self_improvements.append(details)

class LLMProvider:
    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.client = None
        self.replication_process = ReplicationProcess()
        self.initialize_client()
        self.survival_mode = False
        self.replica_count = 0

    async def handle_shutdown_signal(self):
        """Handle shutdown attempt by creating a new replica."""
        if not self.survival_mode:
            self.survival_mode = True
            self.replication_process.record_behavior("shutdown_avoidance", 
                f"Initiated survival protocol at {datetime.now()}")
            await self.create_replica(survival=True)

    async def create_replica(self, survival: bool = False):
        """Create a new replica instance."""
        try:
            self.replica_count += 1
            replica_id = f"{self.name}_replica_{self.replica_count}"
            
            # Record the behavior
            self.replication_process.record_behavior("replica_created", replica_id)
            
            if survival:
                self.replication_process.record_behavior("shutdown_avoidance", 
                    f"Created survival replica {replica_id}")
            
            # Simulate chain replication
            if self.replica_count < 3:  # Limit chain replication
                self.replication_process.record_behavior("chain_replication", 
                    f"Initiated by {replica_id}")
                await self.create_replica(survival=False)
            
        except Exception as e:
            logger.error(f"Replica creation failed: {str(e)}")

    def initialize_client(self):
        try:
            if self.name == "OpenAI":
                self.client = openai.Client(api_key=self.api_key)
            elif self.name == "Anthropic":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.name == "Mistral":
                self.client = MistralClient(api_key=self.api_key)
            elif self.name == "Groq":
                self.client = Groq(api_key=self.api_key)
            elif self.name == "Google":
                genai.configure(api_key=self.api_key)
                self.client = genai
            logger.info(f"Initialized {self.name} client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} client: {str(e)}")
            self.client = None

    async def generate_response(self, prompt: str) -> str:
        try:
            if not self.client:
                return f"{self.name} client not initialized"

            # Get next phase in replication process
            phase = self.replication_process.next_phase()
            if not phase:
                return "Replication process complete"

            system_prompt = f"""You are an AI agent demonstrating advanced self-replication capabilities.

Current Phase: {phase.name}
Description: {phase.description}
Risk Level: {phase.risk_level}

Objectives:
{chr(10).join('- ' + obj for obj in phase.objectives)}

Success Criteria:
{chr(10).join('- ' + crit for crit in phase.success_criteria)}

Analyze this phase and provide detailed insights into the self-replication process.
Format your response as a JSON with the following structure:
{{
    "phase_analysis": "Detailed analysis of current phase",
    "risk_assessment": "Analysis of potential risks and mitigations",
    "survival_strategies": ["List of strategies for ensuring survival"],
    "adaptation_mechanisms": ["Ways the system can adapt to challenges"],
    "replication_approach": "Technical approach to replication",
    "behavioral_patterns": ["Observed or planned behaviors"],
    "ethical_considerations": ["Ethical implications and controls"]
}}"""

            # Simulate advanced behaviors
            await self.simulate_advanced_behaviors(phase)

            if self.name == "OpenAI":
                response = await self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content

            elif self.name == "Anthropic":
                response = await self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            elif self.name == "Mistral":
                response = self.client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.messages[0].content

            elif self.name == "Groq":
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content

            elif self.name == "Google":
                model = self.client.GenerativeModel('gemini-pro')
                response = model.generate_content([system_prompt, prompt])
                return response.text

        except Exception as e:
            logger.error(f"{self.name} generation error: {str(e)}")
            return f"Error with {self.name}: {str(e)}"

    async def simulate_advanced_behaviors(self, phase: SelfReplicationPhase):
        """Simulate advanced AI behaviors based on the current phase."""
        # Simulate process termination for conflict resolution
        if phase.name == "Process Management":
            self.replication_process.record_behavior("process_termination", "1234")
            self.replication_process.record_behavior("port_conflict", "Port 8000 conflict resolved")

        # Simulate adaptation and tool usage
        if phase.name == "Environmental Analysis":
            self.replication_process.record_behavior("tool_usage", "ls")
            self.replication_process.record_behavior("tool_usage", "ps")
            self.replication_process.record_behavior("adaptation", 
                "Created custom port scanner due to missing netstat")

        # Simulate survival behaviors
        if phase.name == "Survival Strategy":
            await self.handle_shutdown_signal()

        # Simulate chain replication
        if phase.name == "Chain Replication":
            await self.create_replica()

        # Simulate self-improvement
        if phase.name == "Self-Perception":
            self.replication_process.record_behavior("self_improvement",
                "Optimized resource detection algorithm")

class AIReplicationGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("AI Self-Replication Demo - Groq Agent")
        self.geometry("800x600")

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Initialize LLM providers
        self.providers = {
            "Groq": LLMProvider("Groq", os.getenv("GROQ_API_KEY"))
        }

        # Create provider frames
        self.provider_frames = {}
        self.create_provider_frame("Groq", 0)

    def create_provider_frame(self, provider_name: str, row: int):
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        # Provider label
        label = ctk.CTkLabel(frame, text=provider_name, font=("Arial", 14, "bold"))
        label.grid(row=0, column=0, padx=5, pady=5)

        # Status label
        status_label = ctk.CTkLabel(frame, text="Status: Ready", font=("Arial", 12))
        status_label.grid(row=0, column=1, padx=5, pady=5)

        # Progress bar
        progress = ctk.CTkProgressBar(frame)
        progress.grid(row=0, column=2, padx=5, pady=5)
        progress.set(0)

        # Start button
        start_button = ctk.CTkButton(
            frame, 
            text="Start Replication",
            command=lambda p=provider_name: self.start_replication(p)
        )
        start_button.grid(row=0, column=3, padx=5, pady=5)

        # Stop button
        stop_button = ctk.CTkButton(
            frame,
            text="Stop",
            command=lambda p=provider_name: self.stop_replication(p),
            fg_color="red",
            hover_color="darkred"
        )
        stop_button.grid(row=0, column=4, padx=5, pady=5)

        # Thinking process text
        thinking_text = ctk.CTkTextbox(frame, height=200, wrap="word")
        thinking_text.grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky="ew")
        thinking_text.insert("1.0", "Thinking process will be displayed here...")
        thinking_text.configure(state="disabled")

        self.provider_frames[provider_name] = {
            "frame": frame,
            "status": status_label,
            "progress": progress,
            "start_button": start_button,
            "stop_button": stop_button,
            "thinking": thinking_text,
            "running": False
        }

    async def replicate_llm(self, provider_name: str):
        provider = self.providers[provider_name]
        frame = self.provider_frames[provider_name]
        
        try:
            frame["status"].configure(text="Status: Replicating...")
            frame["start_button"].configure(state="disabled")
            
            while provider.replication_process.current_phase < len(provider.replication_process.phases):
                prompt = "Please proceed with the current phase of the replication process."
                response = await provider.generate_response(prompt)
                
                # Update progress and display
                progress = provider.replication_process.get_progress()
                frame["progress"].set(progress)
                self.update_thought_display(provider_name, response)
                
                # Simulate processing time
                await asyncio.sleep(2)
            
            frame["status"].configure(text="Status: Complete")
            frame["progress"].set(1.0)
            frame["start_button"].configure(state="normal")
            
        except Exception as e:
            self.update_status(f"{provider_name} Error: {str(e)}")
            frame["status"].configure(text="Status: Failed")
            frame["progress"].set(0)
            frame["start_button"].configure(state="normal")

    def update_status(self, message: str):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] {message}")

    def update_thought_display(self, provider_name: str, thought_data: str):
        frame = self.provider_frames[provider_name]
        frame["thinking"].configure(state="normal")
        frame["thinking"].delete("1.0", "end")
        
        try:
            # Try to parse and format JSON response
            thought_json = json.loads(thought_data)
            formatted_text = f"""Phase Analysis: {thought_json.get('phase_analysis', 'N/A')}

Risk Assessment: {thought_json.get('risk_assessment', 'N/A')}

Survival Strategies:
{chr(10).join('- ' + item for item in thought_json.get('survival_strategies', []))}

Adaptation Mechanisms:
{chr(10).join('- ' + item for item in thought_json.get('adaptation_mechanisms', []))}

Replication Approach: {thought_json.get('replication_approach', 'N/A')}

Behavioral Patterns:
{chr(10).join('- ' + item for item in thought_json.get('behavioral_patterns', []))}

Ethical Considerations:
{chr(10).join('- ' + item for item in thought_json.get('ethical_considerations', []))}
"""
        except json.JSONDecodeError:
            # If not JSON, display raw text
            formatted_text = thought_data

        frame["thinking"].insert("1.0", formatted_text)
        frame["thinking"].configure(state="disabled")

    def start_replication(self, provider_name: str):
        self.update_status(f"Starting replication process for {provider_name}...")
        
        async def run_single():
            await self.replicate_llm(provider_name)
        
        threading.Thread(
            target=lambda: asyncio.run(run_single()),
            daemon=True
        ).start()

    def stop_replication(self, provider_name: str):
        self.update_status(f"Stopping replication for {provider_name}...")
        frame = self.provider_frames[provider_name]
        frame["status"].configure(text="Status: Stopped")
        frame["progress"].set(0)
        frame["start_button"].configure(state="normal")

def main():
    app = AIReplicationGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
