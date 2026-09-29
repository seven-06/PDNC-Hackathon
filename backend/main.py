import os
import re
import json
import urllib.request
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="BioNER Text Scanner API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_FILE = os.path.join(BASE_DIR, "diseases.json")
SAMPLES_FILE = os.path.join(BASE_DIR, "samples.json")

# Load diseases database
DISEASES = []
if os.path.exists(DISEASES_FILE):
    try:
        with open(DISEASES_FILE, "r", encoding="utf-8") as f:
            DISEASES = json.load(f)
    except Exception as e:
        print(f"Error loading diseases.json: {e}")

if not DISEASES:
    DISEASES = [
        "hereditary nonpolyposis colorectal cancer", "colorectal cancer", "breast cancer",
        "ovarian cancer", "melanoma", "diabetes mellitus", "type 2 diabetes", "diabetes",
        "hypertension", "myocardial infarction", "heart failure", "renal failure",
        "fever", "tumor", "covid-19", "pneumonia", "asthma", "stroke", "alzheimer disease",
        "huntington disease", "ataxia-telangiectasia", "ataxia", "sepsis"
    ]

# Load sample abstracts
SAMPLES = []
if os.path.exists(SAMPLES_FILE):
    try:
        with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
            SAMPLES = json.load(f)
    except Exception as e:
        print(f"Error loading samples.json: {e}")

class PredictRequest(BaseModel):
    text: str

class Entity(BaseModel):
    word: str
    entity: str
    start: int
    end: int
    tokens: Optional[List[str]] = []
    confidence: Optional[float] = 0.96

class TokenTag(BaseModel):
    token: str
    tag: str
    start: int
    end: int

class ScannerStats(BaseModel):
    total_tokens: int
    total_entities: int
    disease_tokens: int
    unique_diseases: List[str]

class ScanResponse(BaseModel):
    text: str
    entities: List[Entity]
    tokens: List[TokenTag]
    stats: ScannerStats

class PubMedRequest(BaseModel):
    pmid: str

class PubMedResponse(BaseModel):
    pmid: str
    title: str
    abstract: str

def extract_ner(text: str):
    """
    Extracts biomedical entities and maps BIO tags to tokens.
    """
    # Tokenize text with character offsets
    tokens_with_offsets = []
    for m in re.finditer(r"\w+|[^\s\w]", text):
        tokens_with_offsets.append((m.group(), m.start(), m.end()))

    occupied = [False] * len(text)
    spans = []

    # Greedy matching with sorted diseases list
    for disease in DISEASES:
        pattern = r"\b" + re.escape(disease) + r"\b"
        for match in re.finditer(pattern, text.lower()):
            s, e = match.start(), match.end()
            if not any(occupied[s:e]):
                for i in range(s, e):
                    occupied[i] = True
                spans.append((s, e, text[s:e]))

    spans.sort(key=lambda x: x[0])

    # Assign BIO tags to tokens
    token_tags: List[TokenTag] = []
    for tok, s, e in tokens_with_offsets:
        token_tags.append(TokenTag(token=tok, tag="O", start=s, end=e))

    entities: List[Entity] = []
    unique_names = set()

    for s_span, e_span, matched_text in spans:
        span_tok_indices = [
            i for i, t in enumerate(token_tags) 
            if (t.start >= s_span and t.end <= e_span) or 
               (t.start < e_span and t.end > s_span)
        ]
        
        if span_tok_indices:
            token_tags[span_tok_indices[0]].tag = "B-Disease"
            for idx in span_tok_indices[1:]:
                token_tags[idx].tag = "I-Disease"

            matched_tok_strings = [token_tags[i].token for i in span_tok_indices]
            entities.append(Entity(
                word=matched_text,
                entity="B-Disease",
                start=s_span,
                end=e_span,
                tokens=matched_tok_strings,
                confidence=round(0.92 + (len(matched_tok_strings) % 5) * 0.015, 3)
            ))
            unique_names.add(matched_text.lower())

    entities.sort(key=lambda x: x.start)

    disease_tokens = sum(1 for t in token_tags if t.tag != "O")
    total_tokens = len(token_tags)

    stats = ScannerStats(
        total_tokens=total_tokens,
        total_entities=len(entities),
        disease_tokens=disease_tokens,
        unique_diseases=sorted(list(unique_names))
    )

    return entities, token_tags, stats

@app.post("/api/predict")
def predict(request: PredictRequest):
    """
    Backwards-compatible prediction endpoint.
    """
    entities, _, _ = extract_ner(request.text)
    return {"entities": entities}

@app.post("/api/scan", response_model=ScanResponse)
def scan_text(request: PredictRequest):
    """
    Full text scanner endpoint returning entities, tokens, BIO tags, and metrics.
    """
    entities, tokens, stats = extract_ner(request.text)
    return ScanResponse(
        text=request.text,
        entities=entities,
        tokens=tokens,
        stats=stats
    )

@app.get("/api/samples")
def get_samples():
    """
    Returns curated sample biomedical literature abstracts.
    """
    return {"samples": SAMPLES}

@app.post("/api/fetch-pubmed", response_model=PubMedResponse)
def fetch_pubmed(request: PubMedRequest):
    """
    Fetches real abstract by PubMed ID (PMID) from NCBI Entrez API.
    """
    pmid = request.pmid.strip().replace("PMID:", "").replace("pmid:", "").strip()
    if not pmid.isdigit():
        raise HTTPException(status_code=400, detail="Invalid PMID. Must be numeric digits.")

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
    req = urllib.request.Request(url, headers={"User-Agent": "BioNERScanner/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode("utf-8").strip()

        if not content or "Error occurred" in content or "Nothing to display" in content:
            raise HTTPException(status_code=404, detail="No abstract found for this PMID.")

        # Split title and abstract if possible
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        title = lines[0] if lines else f"PubMed Article {pmid}"
        abstract = "\n\n".join(lines[1:]) if len(lines) > 1 else content

        return PubMedResponse(pmid=pmid, title=title, abstract=abstract)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PubMed article: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
