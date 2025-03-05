from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PaperInput(BaseModel):
    title: str
    abstract_inverted_index: Dict[str, List[int]]
    journal_display_name: str
    referenced_works: List[str]
    inverted: bool
    abstract: Optional[str] = None


class TopicPrediction(BaseModel):
    """Single topic prediction with ID, label and confidence score"""

    topic_id: int = Field(..., description="Topic identifier")
    topic_label: Optional[str] = Field(None, description="Human-readable topic label")
    topic_score: float = Field(..., description="Confidence score between 0 and 1")


class SinglePaperPrediction(BaseModel):
    """Single paper prediction with list of topic predictions and scores"""

    predictions: List[TopicPrediction] = Field(
        ..., description="List of predictions for the paper"
    )


class HealthCheckResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Service health status")
    model: str = Field(..., description="Model status")


class BatchPaperPredictions(BaseModel):
    """Response schema for topic predictions for papers"""

    predictions: List[List[TopicPrediction]] = Field(
        ..., description="List of predictions for each paper"
    )


class ErrorResponse(BaseModel):
    """Error response model"""

    Error: str = Field(..., description="Error message")
