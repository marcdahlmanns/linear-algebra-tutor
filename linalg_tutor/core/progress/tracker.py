"""Progress tracking system with SQLite database."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ..exercises import Exercise, ExerciseResult

Base = declarative_base()


class ExerciseAttempt(Base):
    """Database model for individual exercise attempts."""

    __tablename__ = "exercise_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exercise_id = Column(String, nullable=False, index=True)
    topic = Column(String, nullable=False, index=True)
    difficulty = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    correct = Column(Boolean, nullable=False)
    time_spent = Column(Float, nullable=True)  # seconds
    hints_used = Column(Integer, default=0)
    user_answer = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<ExerciseAttempt(id={self.id}, exercise={self.exercise_id}, correct={self.correct})>"


class TopicProgress(Base):
    """Database model for progress per topic."""

    __tablename__ = "topic_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String, unique=True, nullable=False)
    exercises_attempted = Column(Integer, default=0)
    exercises_correct = Column(Integer, default=0)
    total_time_spent = Column(Float, default=0.0)
    mastery_level = Column(Float, default=0.0)  # 0-1 scale
    last_practiced = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<TopicProgress(topic={self.topic}, mastery={self.mastery_level:.2f})>"


class ProgressTracker:
    """Track and analyze user progress through the curriculum."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize progress tracker.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            # Use default location in data/progress directory
            data_dir = Path.home() / ".linalg_tutor" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "progress.db")

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        # Create session
        Session_class = sessionmaker(bind=self.engine)
        self.session: Session = Session_class()

    def record_attempt(
        self, result: ExerciseResult, exercise: Exercise, time_spent: Optional[float] = None
    ):
        """Record an exercise attempt.

        Args:
            result: Result of the exercise attempt
            exercise: The exercise that was attempted
            time_spent: Time spent on exercise in seconds
        """
        # Create attempt record
        attempt = ExerciseAttempt(
            exercise_id=exercise.exercise_id,
            topic=exercise.topic,
            difficulty=exercise.difficulty.value,
            correct=result.correct,
            time_spent=time_spent or result.time_spent,
            hints_used=result.hints_used,
            user_answer=str(result.user_answer),  # Convert to string for JSON storage
        )

        self.session.add(attempt)
        self.session.commit()

        # Update topic progress
        self._update_topic_progress(exercise.topic, result.correct, time_spent or 0.0)

    def _update_topic_progress(self, topic: str, correct: bool, time_spent: float):
        """Update progress statistics for a topic.

        Args:
            topic: Topic name
            correct: Whether the attempt was correct
            time_spent: Time spent in seconds
        """
        progress = self.session.query(TopicProgress).filter_by(topic=topic).first()

        if progress is None:
            # Create new progress record with explicit defaults
            progress = TopicProgress(
                topic=topic,
                exercises_attempted=0,
                exercises_correct=0,
                total_time_spent=0.0,
                mastery_level=0.0,
            )
            self.session.add(progress)

        # Update statistics
        progress.exercises_attempted = (progress.exercises_attempted or 0) + 1
        if correct:
            progress.exercises_correct = (progress.exercises_correct or 0) + 1
        progress.total_time_spent = (progress.total_time_spent or 0.0) + time_spent
        progress.last_practiced = datetime.utcnow()

        # Calculate mastery level
        progress.mastery_level = self._calculate_mastery(progress)

        self.session.commit()

    def _calculate_mastery(self, progress: TopicProgress) -> float:
        """Calculate mastery level for a topic.

        Uses a combination of accuracy and recency.

        Args:
            progress: TopicProgress object

        Returns:
            Mastery level between 0 and 1
        """
        if progress.exercises_attempted == 0:
            return 0.0

        # Base mastery on accuracy
        accuracy = progress.exercises_correct / progress.exercises_attempted

        # Apply recency decay (5% per day)
        if progress.last_practiced:
            days_since = (datetime.utcnow() - progress.last_practiced).days
            recency_factor = 0.95 ** days_since
        else:
            recency_factor = 0.0

        # Combine accuracy and recency
        mastery = accuracy * (0.7 + 0.3 * recency_factor)  # Weight current accuracy more

        return min(1.0, max(0.0, mastery))

    def get_topic_progress(self, topic: str) -> Optional[TopicProgress]:
        """Get progress for a specific topic.

        Args:
            topic: Topic name

        Returns:
            TopicProgress object or None if not found
        """
        return self.session.query(TopicProgress).filter_by(topic=topic).first()

    def get_all_topics_progress(self) -> List[TopicProgress]:
        """Get progress for all topics.

        Returns:
            List of TopicProgress objects
        """
        return self.session.query(TopicProgress).all()

    def get_mastery_level(self, topic: str) -> float:
        """Get mastery level for a topic.

        Args:
            topic: Topic name

        Returns:
            Mastery level between 0 and 1
        """
        progress = self.get_topic_progress(topic)
        if progress:
            return progress.mastery_level
        return 0.0

    def recommend_next_topic(self) -> str:
        """Recommend next topic to study based on progress.

        Returns:
            Topic name to study next
        """
        # Get all topics
        all_progress = self.get_all_topics_progress()

        if not all_progress:
            # No progress yet, start with first topic
            return "vectors"

        # Find topic with lowest mastery level
        lowest_mastery = min(all_progress, key=lambda p: p.mastery_level)

        # If lowest mastery is already high, user has mastered all topics
        if lowest_mastery.mastery_level > 0.8:
            # Return least recently practiced
            all_progress.sort(key=lambda p: p.last_practiced or datetime.min)
            return all_progress[0].topic

        return lowest_mastery.topic

    def get_statistics(self) -> dict:
        """Get overall statistics.

        Returns:
            Dictionary with statistics
        """
        all_attempts = self.session.query(ExerciseAttempt).all()
        all_progress = self.get_all_topics_progress()

        total_attempts = len(all_attempts)
        total_correct = sum(1 for a in all_attempts if a.correct)
        total_time = sum(a.time_spent or 0 for a in all_attempts)

        avg_mastery = (
            sum(p.mastery_level for p in all_progress) / len(all_progress)
            if all_progress
            else 0.0
        )

        return {
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "accuracy": total_correct / total_attempts if total_attempts > 0 else 0.0,
            "total_time_hours": total_time / 3600,
            "topics_started": len(all_progress),
            "average_mastery": avg_mastery,
        }

    def close(self):
        """Close database session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
