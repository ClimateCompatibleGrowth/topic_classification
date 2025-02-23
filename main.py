import json
from http.client import HTTPException
from predictor import predict, pred_model
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/health_check")
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


@app.post("/single")
async def single(request: Request) -> JSONResponse:
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
        input_json = await request.json()
        if not isinstance(input_json, list):
            input_json = json.loads(input_json)

        if len(input_json) > 1:
            return JSONResponse(
                status_code=400,
                content={"Error": "Only one paper can be processed at a time"},
            )

        all_tags = predict(input_json)
        return JSONResponse(content=all_tags)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch(request: Request) -> JSONResponse:
    """Process a batch of academic papers and return topic predictions.

    Args:
        request: FastAPI Request object containing JSON payload of papers

    Returns:
        JSONResponse with predictions for each paper

    Raises:
        HTTPException: If invalid input or processing error occurs
    """
    try:
        input_json = await request.json()
        if not isinstance(input_json, list):
            raise HTTPException(
                status_code=400, detail="Input must be a list of papers"
            )

        if len(input_json) > 1000:
            raise HTTPException(
                status_code=400, detail="Batch size exceeds limit of 1000 papers"
            )

        if not input_json:
            return JSONResponse(content=[])

        try:
            predictions = predict(input_json)
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
