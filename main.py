"""FastAPI entrypoint exposing the QA service."""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from src.rag import get_answer as fetch_qa_answer
from src.utils import get_shared_logger


logger = get_shared_logger(__name__)

app = FastAPI(
	title="Simple NLP Q&A Service",
	description="Answers questions about member data using a RAG system.",
)


class QuestionIn(BaseModel):
	question: str


class AnswerOut(BaseModel):
	answer: str


@app.post(
	"/ask",
	response_model=AnswerOut,
	summary="Ask a question about the member data",
)
async def ask_question(request: Request, payload: QuestionIn) -> AnswerOut:
	try:
		question = (payload.question or "").strip()
		if not question:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Question cannot be empty.",
			)

		answer_text = fetch_qa_answer(question)
		return AnswerOut(answer=answer_text)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(exc),
		) from exc
	except EnvironmentError as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(exc),
		) from exc
	except RuntimeError as exc:
		raise HTTPException(
			status_code=status.HTTP_502_BAD_GATEWAY,
			detail=str(exc),
		) from exc
	except Exception as exc:  # pragma: no cover - defensive catch
		logger.exception("Unhandled error while processing question.")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="An internal error occurred.",
		) from exc


# Allow cross-origin requests; adjust origins as needed per deployment.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/", summary="Health check")
async def read_root() -> Dict[str, str]:
	return {"status": "ok", "message": "Q&A Service is running."}
