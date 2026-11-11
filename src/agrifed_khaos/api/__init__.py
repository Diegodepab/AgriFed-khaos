"""AgriFed-KHAOS REST API Routers Package."""

from agrifed_khaos.api.catalog import router as catalog_router
from agrifed_khaos.api.datasets import router as datasets_router
from agrifed_khaos.api.hitl import router as hitl_router
from agrifed_khaos.api.inference import router as inference_router
from agrifed_khaos.api.jobs import router as jobs_router
from agrifed_khaos.api.mlops import router as mlops_router
from agrifed_khaos.api.views import router as views_router

__all__ = [
    "catalog_router",
    "datasets_router",
    "hitl_router",
    "inference_router",
    "jobs_router",
    "mlops_router",
    "views_router",
]

