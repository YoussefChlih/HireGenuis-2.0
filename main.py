"""
HireGenius NLP Backend — FastAPI
Scoring hybride CV/JD : 60% SBERT + 40% keyword overlap
Support PDF, DOCX, TXT | Français & Anglais
"""
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Any, Optional

import docx
import pdfplumber
import spacy
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Env ────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5433/hiregenius")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "hiregenius_session")
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
SESSION_DURATION_DAYS = int(os.getenv("SESSION_DURATION_DAYS", "7"))
PASSWORD_ITERATIONS = int(os.getenv("PASSWORD_ITERATIONS", "310000"))

# ─── Database ────────────────────────────────────────────────────────────────
engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class MatchResult(Base):
    __tablename__ = "match_results"
    id = Column(Integer, primary_key=True, index=True)
    cv_filename = Column(String(255))
    job_title = Column(String(500), nullable=True)
    job_description_snippet = Column(Text)
    score = Column(Float)
    semantic_score = Column(Float)
    keyword_score = Column(Float)
    matched_skills = Column(Text)  # JSON list
    missing_skills = Column(Text)  # JSON list
    language = Column(String(10))
    gemini_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="candidate")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="draft")
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    cv_filename = Column(String(255), nullable=False)
    cv_text = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="submitted")
    matching_score = Column(Float, nullable=False, default=0.0)
    semantic_score = Column(Float, nullable=False, default=0.0)
    keyword_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ApplicationAIReport(Base):
    __tablename__ = "application_ai_reports"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(
        Integer, ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    recruiter_summary = Column(Text, nullable=True)
    candidate_summary = Column(Text, nullable=True)
    strengths = Column(JSON, nullable=True)
    gaps = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    score_breakdown = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


sbert_model = None
nlp_model = None
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
WEIGHT_SEMANTIC = 0.6
WEIGHT_KEYWORD = 0.4


def run_startup_migrations() -> None:
    """
    Keep existing databases compatible with new schema fields.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    if "users" in existing_tables:
        existing_columns = {column["name"] for column in inspector.get_columns("users")}
        if "role" not in existing_columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE users ADD COLUMN role VARCHAR(30) NOT NULL DEFAULT 'candidate'")
                )
            logger.info("Migration applied: users.role column created.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        run_startup_migrations()
        logger.info("Database tables ready.")
    except Exception as exc:
        logger.warning(f"Could not create DB tables: {exc}")

    if GEMINI_API_KEY:
        try:
            _ = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini configured.")
        except Exception as exc:
            logger.warning(f"Gemini unavailable: {exc}")
    else:
        logger.warning("No GEMINI_API_KEY — Gemini feedback disabled.")

    logger.info("HireGenius backend ready ✓")
    yield


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HireGenius NLP API",
    version="1.1.0",
    description="Matching CV / Job Description using SBERT + keyword overlap",
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Skills dictionary (multilingual) ────────────────────────────────────────
TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "golang",
    "rust", "swift", "kotlin", "php", "scala", "r", "matlab", "julia",
    "html", "css", "react", "angular", "vue", "nextjs", "nodejs", "express",
    "django", "flask", "fastapi", "spring", "laravel", "rails",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "opencv",
    "transformers", "bert", "gpt", "llm", "rag", "langchain",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions", "linux", "bash",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "firebase", "dynamodb",
    "photoshop", "illustrator", "indesign", "figma", "sketch", "xd",
    "autocad", "solidworks", "blender",
    "excel", "powerpoint", "word", "tableau", "power bi", "sap", "salesforce",
    "jira", "confluence", "agile", "scrum", "kanban",
    "leadership", "communication", "teamwork", "problem solving",
    "project management", "time management", "critical thinking",
    "accounting", "budgeting", "forecasting", "financial analysis", "auditing",
    "tax", "bookkeeping", "gaap", "ifrs",
    "patient care", "clinical research", "emr", "ehr", "hipaa",
    "circuit design", "embedded systems", "fpga", "plc", "cad",
    "mechanical design", "structural analysis",
}

FR_SKILL_MAP = {
    "apprentissage automatique": "machine learning",
    "apprentissage profond": "deep learning",
    "traitement du langage naturel": "nlp",
    "vision par ordinateur": "computer vision",
    "base de données": "sql",
    "gestion de projet": "project management",
    "analyse financière": "financial analysis",
    "soins aux patients": "patient care",
    "conception mécanique": "mechanical design",
    "comptabilité": "accounting",
    "travail d'équipe": "teamwork",
    "communication": "communication",
    "leadership": "leadership",
    "résolution de problèmes": "problem solving",
}

EMAIL_REGEX = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")


# ─── Auth utilities ──────────────────────────────────────────────────────────
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_value.encode("utf-8"),
        PASSWORD_ITERATIONS,
    )
    encoded = base64.urlsafe_b64encode(digest).decode("utf-8")
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt_value}${encoded}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_value, expected_hash = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_value.encode("utf-8"),
            int(iterations),
        )
        candidate = base64.urlsafe_b64encode(digest).decode("utf-8")
        return hmac.compare_digest(candidate, expected_hash)
    except Exception:
        return False


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_signup_payload(full_name: str, email: str, password: str, role: Optional[str] = None) -> dict:
    normalized_name = (full_name or "").strip()
    normalized_email = normalize_email(email)

    if len(normalized_name) < 2:
        raise HTTPException(status_code=400, detail="Le nom complet doit contenir au moins 2 caractères.")
    if len(normalized_name) > 120:
        raise HTTPException(status_code=400, detail="Le nom complet est trop long.")
    if not EMAIL_REGEX.match(normalized_email):
        raise HTTPException(status_code=400, detail="Adresse email invalide.")
    if not PASSWORD_REGEX.match(password or ""):
        raise HTTPException(
            status_code=400,
            detail="Le mot de passe doit contenir au moins 8 caractères, une majuscule, une minuscule et un chiffre.",
        )

    normalized_role = (role or "candidate").strip().lower()
    if normalized_role not in {"candidate", "admin_rh"}:
        raise HTTPException(status_code=400, detail="Role invalide.")

    return {
        "full_name": normalized_name,
        "email": normalized_email,
        "password": password,
        "role": normalized_role,
    }


def create_session_for_user(db, user: User) -> str:
    token = secrets.token_urlsafe(32)
    session = AuthSession(
        user_id=user.id,
        token_hash=hash_session_token(token),
        expires_at=datetime.utcnow() + timedelta(days=SESSION_DURATION_DAYS),
    )
    db.add(session)
    db.commit()
    return token


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_DURATION_DAYS * 24 * 60 * 60,
        expires=SESSION_DURATION_DAYS * 24 * 60 * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/", samesite="lax")


def serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def get_current_user(session_token: Optional[str]) -> User:
    if not session_token:
        raise HTTPException(status_code=401, detail="Authentification requise.")

    db = SessionLocal()
    try:
        token_hash = hash_session_token(session_token)
        session = db.query(AuthSession).filter(AuthSession.token_hash == token_hash).first()
        if not session:
            raise HTTPException(status_code=401, detail="Session invalide.")
        if session.expires_at < datetime.utcnow():
            db.delete(session)
            db.commit()
            raise HTTPException(status_code=401, detail="Session expirée.")

        user = db.query(User).filter(User.id == session.user_id, User.is_active.is_(True)).first()
        if not user:
            raise HTTPException(status_code=401, detail="Utilisateur introuvable.")
        return user
    finally:
        db.close()


def get_current_user_from_cookie(session_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)) -> User:
    return get_current_user(session_token)


def require_role(user: User, allowed_roles: set[str]) -> None:
    if user.role not in allowed_roles:
        raise HTTPException(status_code=403, detail="Acces refuse pour ce role.")


def serialize_job(job: Job) -> dict:
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "status": job.status,
        "created_by": job.created_by,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def serialize_application(application: Application, report: Optional[ApplicationAIReport] = None) -> dict:
    payload = {
        "id": application.id,
        "job_id": application.job_id,
        "candidate_id": application.candidate_id,
        "cv_filename": application.cv_filename,
        "status": application.status,
        "matching_score": application.matching_score,
        "semantic_score": application.semantic_score,
        "keyword_score": application.keyword_score,
        "created_at": application.created_at.isoformat() if application.created_at else None,
        "updated_at": application.updated_at.isoformat() if application.updated_at else None,
    }
    if report:
        payload["report"] = {
            "recruiter_summary": report.recruiter_summary,
            "candidate_summary": report.candidate_summary,
            "strengths": report.strengths or [],
            "gaps": report.gaps or [],
            "recommendations": report.recommendations or [],
            "score_breakdown": report.score_breakdown or {},
        }
    return payload


# ─── Text utilities ───────────────────────────────────────────────────────────
def ensure_models_loaded() -> None:
    global sbert_model, nlp_model

    if sbert_model is None:
        logger.info(f"Loading SBERT model ({SBERT_MODEL_NAME})...")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    if nlp_model is None:
        logger.info("Loading spaCy model (en_core_web_sm)...")
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("en_core_web_sm not found, using blank English model")
            nlp_model = spacy.blank("en")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s\.\,\-\/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_text_from_pdf(content: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def extract_text_from_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(content)
    if ext in ("docx", "doc"):
        return extract_text_from_docx(content)
    if ext == "txt":
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Format non supporté : {ext}. Utilisez PDF, DOCX ou TXT.")


def detect_language(text: str) -> str:
    try:
        return detect(text[:500])
    except Exception:
        return "en"


def normalize_fr_skills(text: str) -> str:
    for fr, en in FR_SKILL_MAP.items():
        text = text.replace(fr, en)
    return text


def extract_skills(text: str) -> set:
    normalized = normalize_fr_skills(text.lower())
    found = set()
    for skill in TECH_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, normalized):
            found.add(skill)
    return found


def hybrid_score(cv_text: str, jd_text: str) -> dict:
    ensure_models_loaded()
    cv_clean = clean_text(cv_text)
    jd_clean = clean_text(jd_text)

    cv_emb = sbert_model.encode(cv_clean, normalize_embeddings=True)
    jd_emb = sbert_model.encode(jd_clean, normalize_embeddings=True)
    semantic = float(util.cos_sim(cv_emb, jd_emb))
    semantic = max(0.0, min(1.0, semantic))

    cv_skills = extract_skills(cv_clean)
    jd_skills = extract_skills(jd_clean)
    keyword = len(cv_skills & jd_skills) / len(jd_skills) if jd_skills else 0.0

    final = (WEIGHT_SEMANTIC * semantic) + (WEIGHT_KEYWORD * keyword)
    return {
        "score": round(final * 100, 2),
        "semantic_score": round(semantic * 100, 2),
        "keyword_score": round(keyword * 100, 2),
        "matched_skills": sorted(cv_skills & jd_skills),
        "missing_skills": sorted(jd_skills - cv_skills),
        "cv_skills": sorted(cv_skills),
        "jd_skills": sorted(jd_skills),
    }


def build_precise_recommendations(score_data: dict, language: str) -> list[str]:
    missing = score_data.get("missing_skills", [])[:3]
    matched = score_data.get("matched_skills", [])[:2]

    if language == "fr":
        recommendations = []
        if missing:
            recommendations.append(f"Ajoutez une section 'Compétences clés' avec: {', '.join(missing)}.")
            recommendations.append(f"Intégrez au moins un projet prouvant {missing[0]} avec résultat mesurable.")
        if matched:
            recommendations.append(f"Renforcez vos acquis ({', '.join(matched)}) avec des métriques de performance.")
        recommendations.append("Personnalisez le résumé du CV en reprenant exactement 2 exigences majeures de l'offre.")
        return recommendations[:4]

    recommendations = []
    if missing:
        recommendations.append(f"Add a dedicated 'Core Skills' section including: {', '.join(missing)}.")
        recommendations.append(f"Add at least one quantified project proving {missing[0]}.")
    if matched:
        recommendations.append(f"Strengthen demonstrated skills ({', '.join(matched)}) with measurable outcomes.")
    recommendations.append("Tailor your CV summary to mirror two top requirements from this job description.")
    return recommendations[:4]


def _looks_generic(text: str) -> bool:
    lowered = (text or "").lower()
    generic_markers = [
        "improve your resume",
        "highlight your skills",
        "be more specific",
        "general recommendation",
        "adaptez votre cv",
        "améliorez votre cv",
        "mettez en avant",
    ]
    return any(marker in lowered for marker in generic_markers)


def parse_json_feedback(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def get_gemini_feedback(cv_text: str, jd_text: str, score_data: dict, language: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "text": "",
            "recommendations": build_precise_recommendations(score_data, language),
        }

    lang_instruction = "Réponds en français." if language == "fr" else "Respond in English."
    prompt = f"""
You are an expert HR analyst. Analyze the match between this CV and job description and return STRICT JSON only.

CV (excerpt):
{cv_text[:1500]}

Job Description (excerpt):
{jd_text[:1500]}

Match Score: {score_data['score']:.1f}/100
- Semantic similarity: {score_data['semantic_score']:.1f}/100
- Keyword match: {score_data['keyword_score']:.1f}/100
- Matched skills: {', '.join(score_data['matched_skills'][:10])}
- Missing skills: {', '.join(score_data['missing_skills'][:10])}

{lang_instruction}

Return this exact JSON schema:
{{
  "feedback_summary": "2-3 sentence assessment",
  "strengths": ["3 precise strengths"],
  "gaps": ["2-3 concrete gaps"],
  "recommendations": ["3-4 concrete actions tied to missing skills"],
  "overall_score": integer from 0 to 100,
  "recommendation_level": "STRONG_MATCH|GOOD_MATCH|PARTIAL_MATCH|NO_MATCH"
}}
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw_text = (response.text or "").strip()
        parsed = parse_json_feedback(raw_text)
        if parsed:
            recs = parsed.get("recommendations") if isinstance(parsed.get("recommendations"), list) else []
            recs = [str(item).strip() for item in recs if str(item).strip()]
            if len(recs) < 2 or any(_looks_generic(item) for item in recs):
                recs = build_precise_recommendations(score_data, language)

            feedback_summary = str(parsed.get("feedback_summary", "")).strip()
            if not feedback_summary:
                feedback_summary = raw_text[:400]
            return {
                "text": feedback_summary,
                "recommendations": recs[:4],
                "structured": parsed,
            }

        return {
            "text": raw_text,
            "recommendations": build_precise_recommendations(score_data, language),
        }
    except Exception as exc:
        logger.error(f"Gemini error: {exc}")
        return {
            "text": "",
            "recommendations": build_precise_recommendations(score_data, language),
        }


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "HireGenius NLP API", "version": "1.1.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": bool(sbert_model is not None and nlp_model is not None),
        "auth_ready": True,
    }


@app.post("/api/auth/signup")
def signup(
    response: Response,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("candidate"),
):
    payload = validate_signup_payload(full_name, email, password, role)
    db = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.email == payload["email"]).first()
        if existing_user:
            raise HTTPException(status_code=409, detail="Un compte existe déjà avec cette adresse email.")

        user = User(
            full_name=payload["full_name"],
            email=payload["email"],
            password_hash=hash_password(payload["password"]),
            role=payload["role"],
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_session_for_user(db, user)
        set_auth_cookie(response, token)
        return {"user": serialize_user(user), "message": "Compte créé avec succès."}
    finally:
        db.close()


@app.post("/api/auth/login")
def login(response: Response, email: str = Form(...), password: str = Form(...)):
    normalized_email = normalize_email(email)
    if not EMAIL_REGEX.match(normalized_email):
        raise HTTPException(status_code=400, detail="Adresse email invalide.")
    if not password:
        raise HTTPException(status_code=400, detail="Mot de passe requis.")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == normalized_email, User.is_active.is_(True)).first()
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

        db.query(AuthSession).filter(AuthSession.user_id == user.id).delete()
        db.commit()

        token = create_session_for_user(db, user)
        set_auth_cookie(response, token)
        return {"user": serialize_user(user), "message": "Connexion réussie."}
    finally:
        db.close()


@app.get("/api/auth/me")
def me(session_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    user = get_current_user(session_token)
    return {"user": serialize_user(user)}


@app.post("/api/auth/logout")
def logout(response: Response, session_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    db = SessionLocal()
    try:
        if session_token:
            token_hash = hash_session_token(session_token)
            session = db.query(AuthSession).filter(AuthSession.token_hash == token_hash).first()
            if session:
                db.delete(session)
                db.commit()
        clear_auth_cookie(response)
        return {"message": "Déconnexion réussie."}
    finally:
        db.close()


@app.post("/api/auth/switch-role")
def switch_role(
    role: str = Form(...),
    user: User = Depends(get_current_user_from_cookie),
):
    normalized_role = (role or "").strip().lower()
    if normalized_role not in {"candidate", "admin_rh"}:
        raise HTTPException(status_code=400, detail="Role invalide.")

    if user.role == normalized_role:
        return {"user": serialize_user(user), "message": "Role deja actif."}

    db = SessionLocal()
    try:
        user_to_update = db.query(User).filter(User.id == user.id).first()
        if not user_to_update:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

        user_to_update.role = normalized_role
        db.commit()
        db.refresh(user_to_update)
        return {"user": serialize_user(user_to_update), "message": "Role mis a jour avec succes."}
    finally:
        db.close()


@app.post("/api/match")
async def match_cv(
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
    job_title: Optional[str] = Form(None),
):
    allowed_ext = {"pdf", "docx", "doc", "txt"}
    ext = cv_file.filename.lower().rsplit(".", 1)[-1] if "." in cv_file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté: .{ext}. Utilisez PDF, DOCX ou TXT.",
        )

    if not job_description or len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="La description du poste doit contenir au moins 50 caractères.",
        )

    content = await cv_file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Fichier trop grand (max 10 MB).")

    try:
        cv_text = extract_text_from_file(cv_file.filename, content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le fichier: {str(exc)}")

    if len(cv_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Le CV semble vide ou illisible. Vérifiez le fichier.",
        )

    language = detect_language(cv_text)

    try:
        result = hybrid_score(cv_text, job_description)
    except Exception as exc:
        logger.error(f"Scoring error: {exc}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du score: {str(exc)}")

    feedback_payload = {"text": "", "recommendations": []}
    if GEMINI_API_KEY:
        try:
            feedback_payload = get_gemini_feedback(cv_text, job_description, result, language)
        except Exception as exc:
            logger.warning(f"Gemini feedback skipped: {exc}")

    try:
        db = SessionLocal()
        record = MatchResult(
            cv_filename=cv_file.filename,
            job_title=job_title or "",
            job_description_snippet=job_description[:500],
            score=result["score"],
            semantic_score=result["semantic_score"],
            keyword_score=result["keyword_score"],
            matched_skills=json.dumps(result["matched_skills"]),
            missing_skills=json.dumps(result["missing_skills"]),
            language=language,
            gemini_feedback=feedback_payload.get("text", ""),
        )
        db.add(record)
        db.commit()
        record_id = record.id
        db.close()
    except Exception as exc:
        logger.warning(f"DB save skipped: {exc}")
        record_id = None

    return JSONResponse(
        {
            "id": record_id,
            "cv_filename": cv_file.filename,
            "job_title": job_title or "",
            "language": language,
            "score": result["score"],
            "semantic_score": result["semantic_score"],
            "keyword_score": result["keyword_score"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "cv_skills": result["cv_skills"],
            "jd_skills": result["jd_skills"],
            "gemini_feedback": feedback_payload.get("text", ""),
            "recommendations": feedback_payload.get("recommendations", []),
        }
    )


@app.post("/api/admin/jobs")
def create_job(
    title: str = Form(...),
    description: str = Form(...),
    user: User = Depends(get_current_user_from_cookie),
):
    require_role(user, {"admin_rh"})
    if len(title.strip()) < 3:
        raise HTTPException(status_code=400, detail="Le titre du poste doit contenir au moins 3 caracteres.")
    if len(description.strip()) < 50:
        raise HTTPException(status_code=400, detail="La description doit contenir au moins 50 caracteres.")

    db = SessionLocal()
    try:
        job = Job(title=title.strip(), description=description.strip(), created_by=user.id, status="draft")
        db.add(job)
        db.commit()
        db.refresh(job)
        return {"job": serialize_job(job), "message": "Offre creee."}
    finally:
        db.close()


@app.get("/api/admin/jobs")
def list_admin_jobs(user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"admin_rh"})
    db = SessionLocal()
    try:
        jobs = db.query(Job).filter(Job.created_by == user.id).order_by(Job.created_at.desc()).all()
        return {"jobs": [serialize_job(job) for job in jobs]}
    finally:
        db.close()


@app.post("/api/admin/jobs/{job_id}/publish")
def publish_job(job_id: int, user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"admin_rh"})
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id, Job.created_by == user.id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Offre introuvable.")
        job.status = "published"
        db.commit()
        db.refresh(job)
        return {"job": serialize_job(job), "message": "Offre publiee."}
    finally:
        db.close()


@app.post("/api/admin/jobs/{job_id}/archive")
def archive_job(job_id: int, user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"admin_rh"})
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id, Job.created_by == user.id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Offre introuvable.")
        job.status = "archived"
        db.commit()
        db.refresh(job)
        return {"job": serialize_job(job), "message": "Offre archivee."}
    finally:
        db.close()


@app.get("/api/admin/jobs/{job_id}/applications")
def admin_job_applications(job_id: int, user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"admin_rh"})
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id, Job.created_by == user.id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Offre introuvable.")
        applications = (
            db.query(Application)
            .filter(Application.job_id == job.id)
            .order_by(Application.matching_score.desc())
            .all()
        )
        return {"applications": [serialize_application(app) for app in applications]}
    finally:
        db.close()


@app.get("/api/admin/applications/{application_id}")
def admin_application_detail(application_id: int, user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"admin_rh"})
    db = SessionLocal()
    try:
        application = db.query(Application).filter(Application.id == application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Candidature introuvable.")
        job = db.query(Job).filter(Job.id == application.job_id, Job.created_by == user.id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Candidature introuvable.")
        report = db.query(ApplicationAIReport).filter(ApplicationAIReport.application_id == application.id).first()
        return {"application": serialize_application(application, report)}
    finally:
        db.close()


@app.get("/api/jobs")
def list_published_jobs():
    db = SessionLocal()
    try:
        jobs = db.query(Job).filter(Job.status == "published").order_by(Job.created_at.desc()).all()
        return {"jobs": [serialize_job(job) for job in jobs]}
    finally:
        db.close()


@app.post("/api/applications")
async def submit_application(
    job_id: int = Form(...),
    cv_file: UploadFile = File(...),
    user: User = Depends(get_current_user_from_cookie),
):
    require_role(user, {"candidate"})
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id, Job.status == "published").first()
        if not job:
            raise HTTPException(status_code=404, detail="Offre non disponible.")

        existing_application = db.query(Application).filter(
            Application.job_id == job_id, Application.candidate_id == user.id
        ).first()
        if existing_application:
            raise HTTPException(status_code=409, detail="Vous avez deja postule a cette offre.")

        allowed_ext = {"pdf", "docx", "doc", "txt"}
        ext = cv_file.filename.lower().rsplit(".", 1)[-1] if "." in cv_file.filename else ""
        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail="Format non supporte. Utilisez PDF, DOCX ou TXT.")

        content = await cv_file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Fichier trop grand (max 10 MB).")

        cv_text = extract_text_from_file(cv_file.filename, content)
        if len(cv_text.strip()) < 100:
            raise HTTPException(status_code=422, detail="Le CV semble vide ou illisible.")

        score_data = hybrid_score(cv_text, job.description)
        language = detect_language(cv_text)
        feedback_payload = get_gemini_feedback(cv_text, job.description, score_data, language)
        feedback_summary = feedback_payload.get("text", "")
        recommendations = feedback_payload.get("recommendations", [])[:4]
        candidate_summary = "Match estime a {score:.1f}%. Concentrez-vous sur: {gaps}".format(
            score=score_data["score"],
            gaps=", ".join(score_data["missing_skills"][:3]) if score_data["missing_skills"] else "renforcer votre impact projet",
        )

        application = Application(
            job_id=job_id,
            candidate_id=user.id,
            cv_filename=cv_file.filename,
            cv_text=cv_text[:10000],
            status="screened",
            matching_score=score_data["score"],
            semantic_score=score_data["semantic_score"],
            keyword_score=score_data["keyword_score"],
        )
        db.add(application)
        db.commit()
        db.refresh(application)

        report = ApplicationAIReport(
            application_id=application.id,
            recruiter_summary=feedback_summary or "Analyse effectuee automatiquement.",
            candidate_summary=candidate_summary,
            strengths=score_data["matched_skills"][:10],
            gaps=score_data["missing_skills"][:10],
            recommendations=recommendations or build_precise_recommendations(score_data, language),
            score_breakdown={
                "semantic": score_data["semantic_score"],
                "keyword": score_data["keyword_score"],
                "overall": score_data["score"],
            },
        )
        db.add(report)
        db.commit()
        return {"application": serialize_application(application, report), "message": "Candidature soumise avec succes."}
    finally:
        db.close()


@app.get("/api/applications/me")
def list_my_applications(user: User = Depends(get_current_user_from_cookie)):
    require_role(user, {"candidate"})
    db = SessionLocal()
    try:
        applications = (
            db.query(Application)
            .filter(Application.candidate_id == user.id)
            .order_by(Application.created_at.desc())
            .all()
        )
        payload = []
        for application in applications:
            report = db.query(ApplicationAIReport).filter(ApplicationAIReport.application_id == application.id).first()
            payload.append(serialize_application(application, report))
        return {"applications": payload}
    finally:
        db.close()


@app.get("/api/history")
def get_history(limit: int = 20):
    try:
        db = SessionLocal()
        results = (
            db.query(MatchResult)
            .order_by(MatchResult.created_at.desc())
            .limit(limit)
            .all()
        )
        db.close()
        return [
            {
                "id": result.id,
                "cv_filename": result.cv_filename,
                "job_title": result.job_title,
                "score": result.score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "language": result.language,
                "created_at": result.created_at.isoformat() if result.created_at else None,
            }
            for result in results
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
