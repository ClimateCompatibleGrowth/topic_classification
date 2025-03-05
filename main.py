from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from predictor import pred_model, predict
from schema import (
    BatchPaperPredictions,
    ErrorResponse,
    HealthCheckResponse,
    PaperInput,
    SinglePaperPrediction,
)

app = FastAPI()


@app.get("/")
@app.get("/health_check", response_model=HealthCheckResponse)
async def health_check():
    """Check if the service and model are healthy and operational.

    Returns:
        JSONResponse with service health status and details

    Raises:
        HTTPException: If model or service is unhealthy
    """
    try:
        if not pred_model:
            raise HTTPException(status_code=503, detail="Model not initialized")

        _ = pred_model.get_layer("output_layer")

        return JSONResponse(
            content={
                "status": "healthy",
                "model": "loaded",
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post(
    "/single",
    response_model=SinglePaperPrediction,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def single(paper_input: List[PaperInput]):
    """Process academic papers and return topic predictions.

    Takes a list of papers with their metadata and returns topic predictions:
    - Validates input JSON format
    - Processes titles and abstracts
    - Analyzes citations
    - Returns topic IDs, labels and confidence scores for each paper

    Args:
        request: FastAPI Request object containing JSON payload

    Returns:
        JSONResponse with list of predictions per paper:
        [
            [
                {
                    "topic_id": int,
                    "topic_label": str,
                    "topic_score": float
                },
                ...
            ],
            ...
        ]

    Raises:
        HTTPException: If invalid JSON or processing error occurs
    """
    try:
        if len(paper_input) > 1:
            return JSONResponse(
                status_code=400,
                content={"Error": "Only one paper can be processed at a time"},
            )

        all_tags = predict([paper_input[0].dict()])
        return JSONResponse(content=all_tags)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchPaperPredictions)
async def batch(paper_inputs: List[PaperInput]):
    """Process a batch of academic papers and return topic predictions.

    Args:
        request: FastAPI Request object containing JSON payload of papers

    Returns:
        JSONResponse with predictions for each paper

    Raises:
        HTTPException: If invalid input or processing error occurs
    """
    try:
        if len(paper_inputs) > 1000:
            raise HTTPException(
                status_code=400, detail="Batch size exceeds limit of 1000 papers"
            )

        if not paper_inputs:
            return JSONResponse(content=[])

        try:
            predictions = predict([paper.dict() for paper in paper_inputs])
            return JSONResponse(content=predictions)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing papers: {str(e)}"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing batch request: {str(e)}"
        )
