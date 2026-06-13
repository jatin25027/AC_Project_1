"""
FastAPI routes for all experiments and bonus tasks.
"""
import asyncio, json, queue as Q
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from api.job_manager import job_manager
from api.experiment_runner import (
    run_rep_analysis, run_model_comparison, run_round_analysis,
    run_confusion_matrix, run_dataset_distribution,
    run_difference_search, run_classical_comparison,
    run_transfer_learning, run_key_recovery,
)
from cipher_implementations.ciphers import get_all_cipher_names

router = APIRouter(prefix="/api")

ALL_CIPHERS = get_all_cipher_names()
ALL_MODELS  = ['MLP', 'CNN', 'SiameseNet', 'MINE']
ALL_REPS    = list(range(1, 11))
REP_LABELS  = {1:"Raw",2:"Diff",3:"Concat",4:"Bit-Slice",5:"Word",
               6:"Intermed",7:"Noisy",8:"Joint P-C",9:"Stats",10:"Sequential"}
CIPHER_META = {
    'skinny':  {'year':2016,'structure':'SPN','block':64},
    'gift64':  {'year':2017,'structure':'SPN','block':64},
    'gift128': {'year':2017,'structure':'SPN','block':128},
    'craft':   {'year':2019,'structure':'SPN','block':64},
    'warp':    {'year':2020,'structure':'GFN','block':128},
    'pipo':    {'year':2020,'structure':'SPN','block':64},
    'ascon':   {'year':2019,'structure':'SPN','block':64},
    'saturnin':{'year':2019,'structure':'SPN','block':64},
    'cham':    {'year':2017,'structure':'ARX','block':64},
    'xoodoo':  {'year':2018,'structure':'SPN','block':64},
    'gimli':   {'year':2017,'structure':'SPN','block':64},
    'sparkle': {'year':2019,'structure':'ARX','block':64},
    'knot':    {'year':2019,'structure':'SPN','block':64},
    'qarma':   {'year':2016,'structure':'SPN','block':64},
}

# ── Meta endpoints ─────────────────────────────────────────────────────────────
@router.get("/ciphers")
def list_ciphers():
    return [{"name": c, **CIPHER_META.get(c, {})} for c in ALL_CIPHERS]

@router.get("/models")
def list_models():
    return ALL_MODELS

@router.get("/representations")
def list_reps():
    return [{"id": k, "label": v} for k, v in REP_LABELS.items()]

# ── SSE progress stream ────────────────────────────────────────────────────────
@router.get("/progress/{job_id}")
async def stream_progress(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    async def gen():
        while True:
            try:
                msg = job.log_queue.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except Q.Empty:
                if job.status in ("done", "error"):
                    break
                yield f"data: {json.dumps({'type':'ping'})}\n\n"
                await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache",
                                      "X-Accel-Buffering":"no"})

@router.get("/results/{job_id}")
def get_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if job.status == "error":
        return JSONResponse({"error": job.error}, status_code=500)
    if job.status != "done":
        return JSONResponse({"status": job.status})
    return {"status": "done", "result": job.result}

# ── Experiment request models ──────────────────────────────────────────────────
class RepRequest(BaseModel):
    ciphers:   List[str] = ['craft']
    models:    List[str] = ['MLP']
    rounds:    List[int] = [3]
    rep_ids:   List[int] = [1,2,3]
    n_samples: int = 2000
    epochs:    int = 8

class ModelRequest(BaseModel):
    ciphers:   List[str] = ['craft']
    models:    List[str] = ['MLP','CNN']
    rounds:    List[int] = [3]
    rep_ids:   List[int] = [2]
    n_samples: int = 2000
    epochs:    int = 8

class RoundRequest(BaseModel):
    ciphers:     List[str] = ['craft']
    models:      List[str] = ['MLP']
    rep_ids:     List[int] = [2]
    round_list:  List[int] = [3,4,5,6]
    n_samples:   int = 2000
    epochs:      int = 8

class CMRequest(BaseModel):
    ciphers:    List[str] = ['craft']
    model_type: str = 'MLP'
    rep_id:     int = 2
    rounds:     int = 3
    n_samples:  int = 2000
    epochs:     int = 8

class DistRequest(BaseModel):
    cipher_name: str = 'craft'
    rounds:      int = 3
    n_samples:   int = 10000

class DiffSearchRequest(BaseModel):
    cipher_name: str = 'skinny'
    rounds:      int = 3
    num_trials:  int = 30

class ClassicalRequest(BaseModel):
    cipher_name: str = 'qarma'
    rounds:      int = 2
    n_samples:   int = 5000

class TransferRequest(BaseModel):
    cipher_name:   str = 'xoodoo'
    source_rounds: int = 3
    target_rounds: int = 4
    rep_id:        int = 2
    n_samples:     int = 2000
    epochs:        int = 8

class KeyRecoveryRequest(BaseModel):
    cipher_name: str = 'gift64'
    rounds:      int = 3
    n_samples:   int = 2000
    epochs:      int = 8

# ── Experiment endpoints ───────────────────────────────────────────────────────
@router.post("/exp/representation")
def exp_representation(req: RepRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_rep_analysis,
        req.ciphers, req.models, req.rounds, req.rep_ids, req.n_samples, req.epochs)
    return {"job_id": jid}

@router.post("/exp/model")
def exp_model(req: ModelRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_model_comparison,
        req.ciphers, req.models, req.rounds, req.rep_ids, req.n_samples, req.epochs)
    return {"job_id": jid}

@router.post("/exp/round")
def exp_round(req: RoundRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_round_analysis,
        req.ciphers, req.models, req.rep_ids, req.round_list, req.n_samples, req.epochs)
    return {"job_id": jid}

@router.post("/exp/confusion")
def exp_confusion(req: CMRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_confusion_matrix,
        req.ciphers, req.model_type, req.rep_id, req.rounds, req.n_samples, req.epochs)
    return {"job_id": jid}

@router.post("/exp/distribution")
def exp_distribution(req: DistRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_dataset_distribution,
        req.cipher_name, req.rounds, req.n_samples)
    return {"job_id": jid}

@router.post("/bonus/diff-search")
def bonus_diff_search(req: DiffSearchRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_difference_search,
        req.cipher_name, req.rounds, req.num_trials)
    return {"job_id": jid}

@router.post("/bonus/classical")
def bonus_classical(req: ClassicalRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_classical_comparison,
        req.cipher_name, req.rounds, req.n_samples)
    return {"job_id": jid}

@router.post("/bonus/transfer")
def bonus_transfer(req: TransferRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_transfer_learning,
        req.cipher_name, req.source_rounds, req.target_rounds,
        req.rep_id, req.n_samples, req.epochs)
    return {"job_id": jid}

@router.post("/bonus/key-recovery")
def bonus_key_recovery(req: KeyRecoveryRequest):
    jid = job_manager.create_job()
    job_manager.run_in_background(jid, run_key_recovery,
        req.cipher_name, req.rounds, req.n_samples, req.epochs)
    return {"job_id": jid}
