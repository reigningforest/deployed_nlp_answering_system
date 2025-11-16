"""FastAPI entrypoint exposing the QA service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from src.rag import get_answer as fetch_qa_answer
from src.utils import get_shared_logger


logger = get_shared_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Pre-initialize models at startup to speed up first request."""
	try:
		from fastembed import TextEmbedding
		from src.rag.spacy_model import ensure_spacy_model
		from src.rag.service import _get_service
		import yaml
		from pathlib import Path
		
		# Load config
		config_path = Path("config/config.yaml")
		if config_path.exists():
			with open(config_path) as f:
				config = yaml.safe_load(f)
			
			# Pre-initialize FastEmbed
			logger.info("Pre-initializing FastEmbed model...")
			TextEmbedding(model_name=config.get("fast_embed_name", "BAAI/bge-small-en-v1.5"))
			logger.info("FastEmbed model initialized")
			
			# Pre-initialize spaCy model
			logger.info("Pre-initializing spaCy model...")
			ensure_spacy_model(
				model_name=config.get("ner_model", "en_core_web_md"),
				version=config.get("ner_model_version", "3.7.0"),
				storage_dir=config.get("ner_model_storage_dir", "./runtime_models/spacy")
			)
			logger.info("spaCy model initialized")
			
			# Pre-initialize the QAService (loads RetrievalEngine with models)
			logger.info("Pre-initializing QA service...")
			_get_service()
			logger.info("QA service initialized and ready")
	except Exception as e:
		logger.warning(f"Failed to pre-initialize models: {e}")
	
	yield
	
	# Cleanup (if needed)


app = FastAPI(
	title="Simple NLP Q&A Service",
	description="Answers questions about member data using a RAG system.",
	lifespan=lifespan,
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
