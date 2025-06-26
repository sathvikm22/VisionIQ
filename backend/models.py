from pydantic import BaseModel
from typing import List

class ComparisonResult(BaseModel):
    filename: str
    similarity_score: float
    num_differences: int
    processed_image_url: str
    visual_output: str
    visual_label: str

class AnalyzeResponse(BaseModel):
    overall_similarity: float
    processing_time: float
    comparisons: List[ComparisonResult]
    pdf_url: str 