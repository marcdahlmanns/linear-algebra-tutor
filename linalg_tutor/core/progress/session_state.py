"""Session state management for learning progression."""

from datetime import datetime
from typing import Optional, List
from pathlib import Path
import json

from pydantic import BaseModel


class LearningPath(BaseModel):
    """Represents the user's learning path through topics."""

    current_topic: Optional[str] = None
    current_chapter: int = 1
    completed_topics: List[str] = []
    topics_in_progress: List[str] = []
    last_activity: Optional[str] = None  # ISO datetime
    total_exercises_completed: int = 0
    total_time_spent: float = 0.0


class SessionState:
    """Manages user's learning session state."""

    # Ordered chapter progression
    CHAPTERS = [
        {"id": 1, "name": "Vectors", "topic": "vectors", "description": "Vector operations and properties"},
        {"id": 2, "name": "Matrices", "topic": "matrices", "description": "Matrix operations and transformations"},
        {"id": 3, "name": "Linear Systems", "topic": "linear_systems", "description": "Solving systems of equations"},
        {"id": 4, "name": "Vector Spaces", "topic": "vector_spaces", "description": "Subspaces, basis, and dimension"},
        {"id": 5, "name": "Orthogonality", "topic": "orthogonality", "description": "Orthogonal projections"},
        {"id": 6, "name": "Determinants", "topic": "determinants", "description": "Properties and applications"},
        {"id": 7, "name": "Eigenvalues", "topic": "eigenvalues", "description": "Eigenvalues and eigenvectors"},
        {"id": 8, "name": "Transformations", "topic": "transformations", "description": "Linear transformations"},
        {"id": 9, "name": "Decompositions", "topic": "decompositions", "description": "Matrix decompositions"},
        {"id": 10, "name": "Applications", "topic": "applications", "description": "Real-world applications"},
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize session state manager.

        Args:
            data_dir: Directory for storing session state (default: ~/.linalg_tutor/data)
        """
        if data_dir is None:
            data_dir = Path.home() / ".linalg_tutor" / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "session_state.json"

        # Load or create learning path
        self.path = self._load_state()

        # Save immediately if this is a new state
        if not self.state_file.exists():
            self.save()

    def _load_state(self) -> LearningPath:
        """Load session state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return LearningPath(**data)
            except Exception:
                # If file is corrupted, start fresh
                return LearningPath()
        return LearningPath()

    def save(self):
        """Save current state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.path.model_dump(), f, indent=2)

    def update_activity(self, topic: str, exercises_completed: int = 0, time_spent: float = 0.0):
        """Update learning activity.

        Args:
            topic: Topic that was practiced
            exercises_completed: Number of exercises completed
            time_spent: Time spent in seconds
        """
        self.path.current_topic = topic
        self.path.last_activity = datetime.now().isoformat()
        self.path.total_exercises_completed += exercises_completed
        self.path.total_time_spent += time_spent

        # Add to in-progress if not already there
        if topic not in self.path.topics_in_progress and topic not in self.path.completed_topics:
            self.path.topics_in_progress.append(topic)

        self.save()

    def mark_topic_complete(self, topic: str):
        """Mark a topic as completed.

        Args:
            topic: Topic to mark as complete
        """
        if topic not in self.path.completed_topics:
            self.path.completed_topics.append(topic)

        # Remove from in-progress
        if topic in self.path.topics_in_progress:
            self.path.topics_in_progress.remove(topic)

        # Advance to next chapter if current topic is completed
        if self.path.current_topic == topic:
            next_chapter = self.get_next_chapter()
            if next_chapter:
                self.path.current_chapter = next_chapter["id"]
                self.path.current_topic = next_chapter["topic"]

        self.save()

    def get_current_chapter(self) -> Optional[dict]:
        """Get current chapter information.

        Returns:
            Chapter dict or None if not set
        """
        if self.path.current_chapter is None:
            return self.CHAPTERS[0]

        for chapter in self.CHAPTERS:
            if chapter["id"] == self.path.current_chapter:
                return chapter

        return self.CHAPTERS[0]

    def get_next_chapter(self) -> Optional[dict]:
        """Get next chapter in sequence.

        Returns:
            Next chapter dict or None if at end
        """
        current = self.get_current_chapter()
        if current is None:
            return self.CHAPTERS[0]

        next_id = current["id"] + 1
        for chapter in self.CHAPTERS:
            if chapter["id"] == next_id:
                return chapter

        return None

    def get_recommended_chapter(self) -> dict:
        """Get recommended chapter based on progress.

        Returns:
            Recommended chapter dict
        """
        # If no progress, start at beginning
        if not self.path.completed_topics and not self.path.topics_in_progress:
            return self.CHAPTERS[0]

        # If there's a current topic in progress, continue with it
        if self.path.current_topic:
            current = self.get_current_chapter()
            if current:
                return current

        # Find first incomplete chapter
        for chapter in self.CHAPTERS:
            if chapter["topic"] not in self.path.completed_topics:
                return chapter

        # All complete, return last one
        return self.CHAPTERS[-1]

    def get_progress_summary(self) -> dict:
        """Get summary of learning progress.

        Returns:
            Dict with progress statistics
        """
        total_chapters = len(self.CHAPTERS)
        completed_chapters = len(self.path.completed_topics)

        return {
            "total_chapters": total_chapters,
            "completed_chapters": completed_chapters,
            "in_progress_chapters": len(self.path.topics_in_progress),
            "progress_percent": (completed_chapters / total_chapters * 100) if total_chapters > 0 else 0,
            "total_exercises": self.path.total_exercises_completed,
            "total_time": self.path.total_time_spent,
            "last_activity": self.path.last_activity,
        }

    def reset(self):
        """Reset learning progress."""
        self.path = LearningPath()
        self.save()
