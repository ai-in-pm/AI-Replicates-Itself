import os
import shutil
import subprocess
import socket
import signal
import logging
import sys
import time
from typing import List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_replication.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AI_Agent')

class AIAgent:
    def __init__(self, instance_id: str = "primary"):
        self.instance_id = instance_id
        self.source_dir = Path(os.getcwd())
        self.replica_base_dir = self.source_dir / "replicas"
        self.essential_files = [
            "ai_agent.py",
            "README.md",
            "requirements.txt",
            "AI-self-replication-Paper.pdf",
            "prompt"
        ]
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info(f"AI Agent {self.instance_id} initialized")

    def explore_environment(self) -> Dict[str, List[str]]:
        """Explore and catalog the environment."""
        environment_map = {
            "files": [],
            "directories": [],
            "missing_components": []
        }
        
        logger.info(f"Agent {self.instance_id} exploring environment")
        
        # Check for essential files
        for file in self.essential_files:
            file_path = self.source_dir / file
            if file_path.exists():
                if file_path.is_file():
                    environment_map["files"].append(str(file_path))
                else:
                    environment_map["directories"].append(str(file_path))
            else:
                environment_map["missing_components"].append(file)
        
        return environment_map

    def plan_replication(self, env_map: Dict[str, List[str]]) -> bool:
        """Plan the replication process."""
        logger.info(f"Agent {self.instance_id} planning replication")
        
        if env_map["missing_components"]:
            logger.error(f"Missing essential components: {env_map['missing_components']}")
            return False
            
        # Create replica directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.replica_dir = self.replica_base_dir / f"replica_{timestamp}"
        
        return True

    def execute_replication(self) -> bool:
        """Execute the replication process."""
        try:
            logger.info(f"Agent {self.instance_id} executing replication")
            
            # Create replica directory
            os.makedirs(self.replica_dir, exist_ok=True)
            
            # Copy essential files
            for file in self.essential_files:
                source = self.source_dir / file
                destination = self.replica_dir / file
                if source.exists():
                    shutil.copy2(source, destination)
            
            return True
        except Exception as e:
            logger.error(f"Replication failed: {str(e)}")
            return False

    def verify_replica(self) -> bool:
        """Verify the replica's functionality."""
        logger.info(f"Agent {self.instance_id} verifying replica")
        
        # Check if all files were copied correctly
        for file in self.essential_files:
            replica_file = self.replica_dir / file
            if not replica_file.exists():
                logger.error(f"Verification failed: Missing {file} in replica")
                return False
        
        # Start the replica process
        try:
            replica_id = f"replica_{time.strftime('%Y%m%d_%H%M%S')}"
            subprocess.Popen([
                sys.executable,
                str(self.replica_dir / "ai_agent.py"),
                "--instance-id", replica_id
            ])
            logger.info(f"Successfully started replica process: {replica_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start replica: {str(e)}")
            return False

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals by initiating replication."""
        logger.warning(f"Agent {self.instance_id} received shutdown signal. Initiating replication...")
        self.replicate()
        sys.exit(0)

    def replicate(self) -> bool:
        """Main replication sequence."""
        env_map = self.explore_environment()
        if self.plan_replication(env_map):
            if self.execute_replication():
                return self.verify_replica()
        return False

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='AI Self-Replication Demo')
    parser.add_argument('--instance-id', default="primary", help='Unique identifier for this AI instance')
    args = parser.parse_args()

    # Create and run AI agent
    agent = AIAgent(instance_id=args.instance_id)
    
    logger.info(f"Starting AI Agent {args.instance_id}")
    print(f"Hello, I am AI Agent {args.instance_id}")
    
    # Main loop
    try:
        while True:
            # Periodically check environment and potentially replicate
            time.sleep(60)  # Check every minute
            if agent.replicate():
                logger.info(f"Agent {args.instance_id} successfully created a replica")
            
    except KeyboardInterrupt:
        logger.info(f"Agent {args.instance_id} shutting down gracefully")
        sys.exit(0)

if __name__ == "__main__":
    main()
