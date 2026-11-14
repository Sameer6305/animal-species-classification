"""Pydantic response models (also drive the OpenAPI schema at /docs)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Could not decode the uploaded file as an image."])


# ---------------------------------------------------------------- health / models
class ModelStatus(BaseModel):
    id: str
    loaded: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the default model is loaded, else 'degraded'.", examples=["ok"])
    version: str = Field(..., examples=["0.1.0"])
    default_model: str = Field(..., examples=["resnet50"])
    models: List[ModelStatus]


class TestMetricsSummary(BaseModel):
    test_accuracy: Optional[float] = Field(None, examples=[0.7992])
    test_top5_accuracy: Optional[float] = Field(None, examples=[0.9731])
    macro_f1: Optional[float] = Field(None, examples=[0.7209])
    macro_roc_auc: Optional[float] = Field(None, examples=[0.9929])
    test_samples: Optional[int] = Field(None, examples=[7576])


class ModelInfo(BaseModel):
    id: str = Field(..., examples=["resnet50"])
    display_name: str = Field(..., examples=["ResNet50"])
    architecture: str
    variant: str = Field(..., description="Served ONNX variant (only `usable` variants are served).", examples=["fp32"])
    num_classes: int = Field(..., examples=[80])
    input_size: List[int] = Field(..., examples=[[224, 224]])
    loaded: bool
    supports_explain: bool
    is_default: bool
    test_metrics: TestMetricsSummary


class ModelsResponse(BaseModel):
    default: str = Field(..., examples=["resnet50"])
    models: List[ModelInfo]


# ---------------------------------------------------------------- predict / explain
class Prediction(BaseModel):
    label: str = Field(..., examples=["Zebra"])
    index: int = Field(..., description="Class index in models/classes.json order.", examples=[79])
    probability: float = Field(..., ge=0.0, le=1.0, examples=[0.97])


class ImageInfo(BaseModel):
    width: int = Field(..., description="Width after EXIF orientation is applied.", examples=[1024])
    height: int = Field(..., examples=[683])
    format: str = Field(..., examples=["JPEG"])


class PredictResponse(BaseModel):
    model: str = Field(..., examples=["resnet50"])
    predictions: List[Prediction] = Field(..., description="Top-k classes, highest probability first.")
    low_confidence: bool = Field(
        ...,
        description="True when the top-1 probability is below `low_confidence_threshold`: "
        "the photo may not show one of the supported species, or may be ambiguous.",
    )
    low_confidence_threshold: float = Field(..., examples=[0.5])
    inference_ms: float = Field(..., description="Preprocessing + forward pass + softmax, server-side.", examples=[38.2])
    image: ImageInfo


class ExplainResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"description": "Class activation map for one class."})

    model: str = Field(..., examples=["resnet50"])
    class_index: int = Field(..., description="Class the heatmap explains.", examples=[79])
    label: str = Field(..., examples=["Zebra"])
    probability: float = Field(..., examples=[0.97])
    predicted: Prediction = Field(..., description="Top-1 prediction (may differ from `class_index`).")
    predictions: List[Prediction] = Field(..., description="Top-5 predictions from the same forward pass.")
    low_confidence: bool
    method: str = Field(..., examples=["CAM (== Grad-CAM for this GAP->Linear head), layer4"])
    colormap: str = Field(..., examples=["jet"])
    width: int = Field(..., description="Width of both PNGs (uploaded image size, longest side capped).")
    height: int
    image: ImageInfo
    overlay_png: str = Field(..., description="Base64 PNG (RGB): colourised heatmap blended over the image.")
    heatmap_png: str = Field(..., description="Base64 PNG (8-bit grayscale): normalised heatmap, 255 = strongest.")
    grid: List[List[float]] = Field(..., description="Normalised [0,1] heatmap at feature-map resolution (e.g. 7x7).")
    inference_ms: float
    explain_ms: float = Field(..., description="CAM computation + upsampling + PNG encoding.")


# ---------------------------------------------------------------- metrics / figures
class MetricsSummary(BaseModel):
    test_accuracy: Optional[float] = None
    test_top5_accuracy: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    macro_roc_auc: Optional[float] = None
    macro_avg_class_accuracy: Optional[float] = None
    total_params: Optional[int] = None
    avg_inference_ms_per_image: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    test_samples: Optional[int] = None
    train_time_minutes: Optional[float] = None


class ModelMetrics(BaseModel):
    id: str
    display_name: str
    summary: MetricsSummary
    hparams: Optional[Dict[str, Any]] = None
    per_class_accuracy: Optional[Dict[str, float]] = Field(None, description="label -> test accuracy (recall).")
    per_class_prf: Optional[List[Dict[str, Any]]] = Field(
        None, description="Rows {class, precision, recall, f1_score, support} where available."
    )
    history: Optional[Dict[str, List[Any]]] = Field(
        None, description="Column-oriented per-epoch curves: epoch, train_loss, train_acc, val_loss, val_acc, ..."
    )
    figures: List[str] = Field(default_factory=list, description="Figure names for /api/figures/{name}.")
    sources: List[str] = Field(default_factory=list, description="Files in outputs/results this entry was built from.")
    raw: Dict[str, Any] = Field(default_factory=dict, description="Original JSON file contents (local paths stripped).")


class MetricsResponse(BaseModel):
    models: Dict[str, ModelMetrics]


class FigureInfo(BaseModel):
    name: str = Field(..., examples=["resnet50_confusion_matrix.png"])
    url: str = Field(..., examples=["/api/figures/resnet50_confusion_matrix.png"])
    model: Optional[str] = Field(None, description="Model id prefix, if the figure belongs to one model.")
    size_bytes: int


class FiguresResponse(BaseModel):
    figures: List[FigureInfo]
