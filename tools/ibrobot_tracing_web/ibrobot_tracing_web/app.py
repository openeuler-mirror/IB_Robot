"""FastAPI application factory and versioned tracing API routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import Response as FastAPIResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ibrobot_tracing.critical_path import (
    DEFAULT_MAX_CRITICAL_PATH_SEGMENTS,
    MAX_CRITICAL_PATH_SEGMENTS,
)
from ibrobot_tracing.projection import (
    DEFAULT_BUCKET_REQUEST_LIMIT,
    DEFAULT_DISTRIBUTION_BINS,
    DEFAULT_DISTRIBUTION_OUTLIERS,
    MAX_BUCKET_REQUEST_LIMIT,
    MAX_DISTRIBUTION_BINS,
    MAX_DISTRIBUTION_OUTLIERS,
    MIN_BUCKET_REQUEST_LIMIT,
    MIN_DISTRIBUTION_BINS,
    MIN_DISTRIBUTION_OUTLIERS,
)

from .catalog import CatalogSnapshot, SourceCatalog, SourceChangedError, SourceNotFoundError, UnsafeSourceError
from .config import WebConfig, validate_lan_exposure
from .jobs import (
    AnalysisEntry,
    AnalysisManager,
    AnalysisNotFoundError,
    ComparisonEntry,
    ComparisonJobNotFoundError,
    ComparisonJobRecord,
    ComparisonManager,
    ComparisonNotFoundError,
    ComparisonQueueFullError,
    ComparisonSourceUnavailableError,
    JobNotFoundError,
    JobRecord,
    QueueFullError,
    SameSourceComparisonError,
    TracingAnalyzer,
    analysis_counts,
)
from .queries import ResultQueries
from .schemas import (
    Analysis,
    AnalysisJob,
    AnalysisJobCreate,
    AnalysisListResponse,
    CallTreeResponse,
    CapabilitiesResponse,
    ComparisonJob,
    ComparisonJobCreate,
    ComparisonListResponse,
    ComparisonResultResponse,
    ComparisonSummary,
    ComponentListResponse,
    CriticalPathResponse,
    EventPage,
    FlowPage,
    GraphResponse,
    HealthResponse,
    LatencyDistributionResponse,
    Page,
    Source,
    SourceListResponse,
    SpanPage,
    SpanProfileResponse,
    SummaryResponse,
    TimelineResponse,
    TracepointPage,
    WarningListResponse,
)

Offset = Annotated[int, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=1000)]
FilterText = Annotated[str, Query(max_length=256)]
OptionalMetric = Annotated[str | None, Query(min_length=1, max_length=256)]
MAX_SPAN_PROFILE_NODES = 10_000
SpanProfileNodeLimit = Annotated[int, Query(ge=1, le=MAX_SPAN_PROFILE_NODES)]
CriticalPathSegmentLimit = Annotated[int, Query(ge=1, le=MAX_CRITICAL_PATH_SEGMENTS)]
RequiredRequestId = Annotated[str, Query(min_length=1, max_length=256)]
DistributionBins = Annotated[int, Query(ge=MIN_DISTRIBUTION_BINS, le=MAX_DISTRIBUTION_BINS)]
DistributionOutlierLimit = Annotated[
    int,
    Query(ge=MIN_DISTRIBUTION_OUTLIERS, le=MAX_DISTRIBUTION_OUTLIERS),
]
BucketRequestLimit = Annotated[int, Query(ge=MIN_BUCKET_REQUEST_LIMIT, le=MAX_BUCKET_REQUEST_LIMIT)]
RequestSort = Literal[
    "action_chunk_publish_ms",
    "cloud_roundtrip_ms",
    "dispatch_decode_ms",
    "dispatch_to_infer_ms",
    "execute_publish_ms",
    "inference_ms",
    "obs_frame_ms",
    "policy_total_reported_ms",
    "postprocess_ms",
    "preprocess_ms",
    "queue_refill_ms",
    "refill_to_execute_ms",
    "request_id",
    "total_ms",
]


def _source(source: Any) -> Source:
    return Source(
        id=source.source_id,
        name=source.name,
        kind=source.kind,
        version=source.fingerprint,
        modified_at=source.modified_at,
        size_bytes=source.size_bytes,
        file_count=source.file_count,
    )


def _source_list(snapshot: CatalogSnapshot) -> SourceListResponse:
    return SourceListResponse(
        generation=snapshot.generation,
        refreshed_at=snapshot.refreshed_at,
        items=[_source(source) for source in snapshot.sources],
        warnings=list(snapshot.warnings),
    )


def _job(job: JobRecord, *, queue_position: int | None = None, deduplicated: bool = False) -> AnalysisJob:
    return AnalysisJob(
        id=job.job_id,
        source_id=job.source_id,
        source_version=job.source_version,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        analysis_id=job.analysis_id,
        error=job.error,
        cancel_requested=job.cancel_requested,
        queue_position=queue_position,
        deduplicated=deduplicated,
    )


def _analysis(entry: AnalysisEntry) -> Analysis:
    return Analysis(
        id=entry.analysis_id,
        source_id=entry.source_id,
        source_name=entry.source_name,
        source_kind=entry.source_kind,
        source_version=entry.source_version,
        created_at=entry.created_at,
        **analysis_counts(entry),
    )


def _comparison_job(
    job: ComparisonJobRecord,
    *,
    queue_position: int | None = None,
    deduplicated: bool = False,
) -> ComparisonJob:
    parameters = job.parameters
    return ComparisonJob(
        id=job.job_id,
        baseline_source_id=job.baseline_source_id,
        baseline_source_version=job.baseline_source_version,
        candidate_analysis_id=job.candidate_analysis_id,
        statistic=parameters.statistic,
        relative_threshold_percent=parameters.relative_threshold_percent,
        absolute_threshold_ms=parameters.absolute_threshold_ms,
        bins=parameters.bins,
        metric=parameters.metric,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        comparison_id=job.comparison_id,
        error=job.error,
        queue_position=queue_position,
        deduplicated=deduplicated,
    )


def _comparison(entry: ComparisonEntry) -> ComparisonResultResponse:
    return ComparisonResultResponse(
        id=entry.comparison_id,
        job_id=entry.job_id,
        baseline_source_id=entry.baseline_source_id,
        baseline_source_version=entry.baseline_source_version,
        candidate_analysis_id=entry.candidate_analysis_id,
        created_at=entry.created_at,
        **entry.result.to_dict(),
    )


def _comparison_summary(entry: ComparisonEntry) -> ComparisonSummary:
    result = entry.result
    return ComparisonSummary(
        id=entry.comparison_id,
        job_id=entry.job_id,
        baseline_source_id=entry.baseline_source_id,
        baseline_source_version=entry.baseline_source_version,
        candidate_analysis_id=entry.candidate_analysis_id,
        created_at=entry.created_at,
        statistic=result.statistic,
        metric=result.histogram.metric,
        comparable=result.comparable,
        has_regression=result.has_regression,
        warning_count=len(result.coverage_warnings) + len(result.count_warnings),
    )


def _entry(manager: AnalysisManager, analysis_id: str) -> AnalysisEntry:
    try:
        return manager.get_analysis(analysis_id)
    except AnalysisNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis was not found") from exc


def _job_record(manager: AnalysisManager, job_id: str) -> tuple[JobRecord, int | None]:
    try:
        return manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job was not found") from exc


def _comparison_job_record(
    manager: ComparisonManager,
    job_id: str,
) -> tuple[ComparisonJobRecord, int | None]:
    try:
        return manager.get_job(job_id)
    except ComparisonJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Comparison job was not found or has expired") from exc


def _comparison_entry(manager: ComparisonManager, comparison_id: str) -> ComparisonEntry:
    try:
        return manager.get_comparison(comparison_id)
    except ComparisonNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Comparison result was not found or has expired") from exc


class _MutationOriginMiddleware:
    def __init__(self, app, *, origins):
        self.app = app
        self.origins = origins

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in {"POST", "DELETE"}:
            request = Request(scope)
            origin = request.headers.get("origin")
            same_origin = f"{request.url.scheme}://{request.url.netloc}"
            if (origin is not None and origin != same_origin and origin not in self.origins) or (
                request.headers.get("sec-fetch-site") == "cross-site" and origin not in self.origins
            ):
                await JSONResponse(status_code=403, content={"detail": "Cross-site mutation is not allowed"})(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def create_app(
    config: WebConfig | None = None,
    *,
    catalog: SourceCatalog | None = None,
    manager: AnalysisManager | None = None,
    comparison_manager: ComparisonManager | None = None,
) -> FastAPI:
    config = config or WebConfig()
    validate_lan_exposure(config)
    catalog = catalog or SourceCatalog(
        config.trace_roots,
        max_sources=config.max_sources,
        max_scan_entries=config.max_scan_entries,
        max_source_bytes=config.max_source_bytes,
    )
    manager = manager or AnalysisManager(
        catalog,
        queue_size=config.queue_size,
        max_analyses=config.max_analyses,
        max_job_history=config.max_job_history,
        analyzer=TracingAnalyzer(config.max_events),
        max_result_bytes=config.max_result_bytes,
    )
    comparison_manager = comparison_manager or ComparisonManager(
        catalog,
        manager,
        queue_size=config.queue_size,
        max_comparisons=config.max_analyses,
        max_job_history=config.max_job_history,
        max_events=config.max_events,
        max_flame_nodes=MAX_SPAN_PROFILE_NODES,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        catalog.refresh()
        manager.start()
        comparison_manager.start()
        try:
            yield
        finally:
            comparison_manager.close()
            manager.close()

    app = FastAPI(
        title="IB-Robot Tracing API",
        version="1.0.0",
        description="Read-only trace catalog with bounded offline analysis and comparison",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.catalog = catalog
    app.state.analysis_manager = manager
    app.state.comparison_manager = comparison_manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.allowed_hosts))
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )

    app.add_middleware(_MutationOriginMiddleware, origins=config.cors_origins)

    router = APIRouter(prefix="/api/v1")

    @app.get("/healthz", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        snapshot = catalog.snapshot()
        worker_running = manager.worker_running and comparison_manager.worker_running
        return HealthResponse(
            status="ok" if worker_running else "degraded",
            worker_running=worker_running,
            source_count=len(snapshot.sources),
            analysis_count=manager.analysis_count,
            queued_jobs=manager.queued_count + comparison_manager.queued_count,
        )

    @router.get("/capabilities", response_model=CapabilitiesResponse, tags=["service"])
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            tls_enabled=config.tls_enabled,
            cors_enabled=bool(config.cors_origins),
            baseline_compare=True,
            source_kinds=["ctf", "log"],
            analysis_views=[
                "summary",
                "distribution",
                "requests",
                "events",
                "spans",
                "flows",
                "components",
                "tracepoints",
                "timeline",
                "call-tree",
                "span-profile",
                "critical-path",
                "graph",
                "warnings",
            ],
            limits={
                "queue_size": config.queue_size,
                "max_analyses": config.max_analyses,
                "max_job_history": config.max_job_history,
                "max_sources": config.max_sources,
                "max_source_bytes": config.max_source_bytes,
                "max_events": config.max_events,
                "max_result_bytes": config.max_result_bytes,
                "max_page_size": 1000,
                "max_span_profile_nodes": MAX_SPAN_PROFILE_NODES,
                "max_critical_path_segments": MAX_CRITICAL_PATH_SEGMENTS,
                "max_distribution_bins": MAX_DISTRIBUTION_BINS,
                "max_distribution_outliers": MAX_DISTRIBUTION_OUTLIERS,
                "max_bucket_request_ids": MAX_BUCKET_REQUEST_LIMIT,
                "comparison_queue_size": config.queue_size,
                "max_comparison_results": config.max_analyses,
                "max_comparison_job_history": config.max_job_history,
                "min_comparison_bins": 5,
                "max_comparison_bins": 100,
                "max_comparison_flame_nodes": MAX_SPAN_PROFILE_NODES,
            },
        )

    @router.get("/sources", response_model=SourceListResponse, tags=["sources"])
    def list_sources() -> SourceListResponse:
        return _source_list(catalog.snapshot())

    @router.post("/sources/refresh", response_model=SourceListResponse, tags=["sources"])
    def refresh_sources() -> SourceListResponse:
        return _source_list(catalog.refresh_coalesced())

    @router.get("/sources/{source_id}", response_model=Source, tags=["sources"])
    def source_detail(source_id: str) -> Source:
        try:
            return _source(catalog.get(source_id))
        except SourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Trace source was not found") from exc

    @router.post("/analysis-jobs", response_model=AnalysisJob, tags=["analysis jobs"])
    def create_analysis_job(request: AnalysisJobCreate, response: Response) -> AnalysisJob:
        try:
            job, deduplicated = manager.submit(request.source_id)
        except SourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Trace source was not found") from exc
        except (SourceChangedError, UnsafeSourceError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
        response.status_code = status.HTTP_200_OK if deduplicated else status.HTTP_202_ACCEPTED
        current, queue_position = _job_record(manager, job.job_id)
        return _job(current, queue_position=queue_position, deduplicated=deduplicated)

    @router.get("/analysis-jobs/{job_id}", response_model=AnalysisJob, tags=["analysis jobs"])
    def analysis_job_detail(job_id: str) -> AnalysisJob:
        job, queue_position = _job_record(manager, job_id)
        return _job(job, queue_position=queue_position)

    @router.post("/analysis-jobs/{job_id}/cancel", response_model=AnalysisJob, tags=["analysis jobs"])
    def cancel_analysis_job(job_id: str) -> AnalysisJob:
        try:
            job, queue_position = manager.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Analysis job was not found") from exc
        return _job(job, queue_position=queue_position)

    @router.post("/comparison-jobs", response_model=ComparisonJob, tags=["comparison jobs"])
    def create_comparison_job(request: ComparisonJobCreate, response: Response) -> ComparisonJob:
        try:
            job, deduplicated = comparison_manager.submit(
                request.baseline_source_id,
                request.baseline_source_version,
                request.candidate_analysis_id,
                statistic=request.statistic,
                relative_threshold_percent=request.relative_threshold_percent,
                absolute_threshold_ms=request.absolute_threshold_ms,
                metric=request.metric,
                bins=request.bins,
            )
        except SourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Baseline trace source was not found") from exc
        except (
            ComparisonSourceUnavailableError,
            SameSourceComparisonError,
            SourceChangedError,
            UnsafeSourceError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AnalysisNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Candidate analysis was not found") from exc
        except ComparisonQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Comparison service is unavailable") from exc
        response.status_code = status.HTTP_200_OK if deduplicated else status.HTTP_202_ACCEPTED
        current, queue_position = _comparison_job_record(comparison_manager, job.job_id)
        return _comparison_job(current, queue_position=queue_position, deduplicated=deduplicated)

    @router.get("/comparison-jobs/{job_id}", response_model=ComparisonJob, tags=["comparison jobs"])
    def comparison_job_detail(job_id: str) -> ComparisonJob:
        job, queue_position = _comparison_job_record(comparison_manager, job_id)
        return _comparison_job(job, queue_position=queue_position)

    @router.get("/comparisons", response_model=ComparisonListResponse, tags=["comparisons"])
    def list_comparisons() -> ComparisonListResponse:
        return ComparisonListResponse(
            items=[_comparison_summary(entry) for entry in comparison_manager.list_comparisons()]
        )

    @router.get("/comparisons/{comparison_id}", response_model=ComparisonResultResponse, tags=["comparisons"])
    def comparison_detail(comparison_id: str) -> ComparisonResultResponse:
        return _comparison(_comparison_entry(comparison_manager, comparison_id))

    @router.delete("/comparisons/{comparison_id}", status_code=204, tags=["comparisons"])
    def delete_comparison(comparison_id: str) -> FastAPIResponse:
        try:
            comparison_manager.delete_comparison(comparison_id)
        except ComparisonNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Comparison result was not found or has expired") from exc
        return FastAPIResponse(status_code=204)

    @router.get("/analyses", response_model=AnalysisListResponse, tags=["analyses"])
    def list_analyses() -> AnalysisListResponse:
        return AnalysisListResponse(items=[_analysis(entry) for entry in manager.list_analyses()])

    @router.get("/analyses/{analysis_id}", response_model=Analysis, tags=["analyses"])
    def analysis_detail(analysis_id: str) -> Analysis:
        return _analysis(_entry(manager, analysis_id))

    @router.delete("/analyses/{analysis_id}", status_code=204, tags=["analyses"])
    def delete_analysis(analysis_id: str) -> FastAPIResponse:
        try:
            manager.delete_analysis(analysis_id)
        except AnalysisNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Analysis was not found") from exc
        return FastAPIResponse(status_code=204)

    @router.get("/analyses/{analysis_id}/summary", response_model=SummaryResponse, tags=["analysis views"])
    def summary(analysis_id: str) -> SummaryResponse:
        entry = _entry(manager, analysis_id)
        queries = ResultQueries(entry.result)
        return SummaryResponse(
            analysis=_analysis(entry),
            metadata=queries.metadata(),
            stages=queries.stage_summary(),
            observations=queries.observations(),
            **queries.span_summary(),
            custom_span_summary=queries.custom_span_summary(),
            custom_mark_summary=queries.custom_mark_summary(),
            coverage=queries.coverage(),
        )

    @router.get("/analyses/{analysis_id}/requests", response_model=Page, tags=["analysis views"])
    def requests(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        sort_by: RequestSort = "total_ms",
        descending: bool = True,
        request_id: FilterText = "",
    ) -> Page:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return Page(
            **queries.query_requests(
                request_id=request_id,
                sort_by=sort_by,
                descending=descending,
                offset=offset,
                limit=limit,
            )
        )

    @router.get(
        "/analyses/{analysis_id}/distribution",
        response_model=LatencyDistributionResponse,
        tags=["analysis views"],
    )
    def latency_distribution(
        analysis_id: str,
        metric: OptionalMetric = None,
        bins: DistributionBins = DEFAULT_DISTRIBUTION_BINS,
        outlier_limit: DistributionOutlierLimit = DEFAULT_DISTRIBUTION_OUTLIERS,
        bucket_request_limit: BucketRequestLimit = DEFAULT_BUCKET_REQUEST_LIMIT,
    ) -> LatencyDistributionResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        try:
            document = queries.latency_distribution_projection(
                metric=metric,
                bins=bins,
                outlier_limit=outlier_limit,
                bucket_request_limit=bucket_request_limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return LatencyDistributionResponse(**document)

    @router.get("/analyses/{analysis_id}/events", response_model=EventPage, tags=["analysis views"])
    def events(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        request_id: FilterText = "",
        component_id: FilterText = "",
        event_name: FilterText = "",
        provider: FilterText = "",
        origin: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> EventPage:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return EventPage(
            **queries.query_events(
                offset=offset,
                limit=limit,
                request_id=request_id,
                component_id=component_id,
                event_name=event_name,
                provider=provider,
                origin_kind=origin,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )

    @router.get("/analyses/{analysis_id}/spans", response_model=SpanPage, tags=["analysis views"])
    def spans(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        request_id: FilterText = "",
        component_id: FilterText = "",
        name: FilterText = "",
        span_status: FilterText = "",
        origin: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> SpanPage:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return SpanPage(
            **queries.query_spans(
                offset=offset,
                limit=limit,
                request_id=request_id,
                component_id=component_id,
                name=name,
                status=span_status,
                origin=origin,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )

    @router.get("/analyses/{analysis_id}/flows", response_model=FlowPage, tags=["analysis views"])
    def flows(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        request_id: FilterText = "",
        component_id: FilterText = "",
        edge_id: FilterText = "",
        flow_status: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> FlowPage:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return FlowPage(
            **queries.query_flows(
                offset=offset,
                limit=limit,
                request_id=request_id,
                component_id=component_id,
                edge_id=edge_id,
                status=flow_status,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )

    @router.get("/analyses/{analysis_id}/components", response_model=ComponentListResponse, tags=["analysis views"])
    def components(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        component_id: FilterText = "",
        parent_id: FilterText = "",
        kind: FilterText = "",
        provenance: FilterText = "",
    ) -> ComponentListResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return ComponentListResponse(
            **queries.query_components(
                offset=offset,
                limit=limit,
                component_id=component_id,
                parent_id=parent_id,
                kind=kind,
                provenance=provenance,
            )
        )

    @router.get("/analyses/{analysis_id}/tracepoints", response_model=TracepointPage, tags=["analysis views"])
    def tracepoints(
        analysis_id: str,
        offset: Offset = 0,
        limit: Limit = 100,
        kind: Literal["event", "span"] | None = None,
        component_id: FilterText = "",
        name: FilterText = "",
        origin: FilterText = "",
    ) -> TracepointPage:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return TracepointPage(
            **queries.query_tracepoints(
                offset=offset,
                limit=limit,
                kind=kind or "",
                component_id=component_id,
                name=name,
                origin=origin,
            )
        )

    @router.get("/analyses/{analysis_id}/timeline", response_model=TimelineResponse, tags=["analysis views"])
    def timeline(
        analysis_id: str,
        request_id: FilterText = "",
        component_id: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
        include_events: bool = True,
        include_spans: bool = True,
        include_flows: bool = True,
    ) -> TimelineResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return TimelineResponse(
            **queries.timeline_projection(
                request_id=request_id,
                component_id=component_id,
                start_ns=start_ns,
                end_ns=end_ns,
                include_events=include_events,
                include_spans=include_spans,
                include_flows=include_flows,
            )
        )

    @router.get("/analyses/{analysis_id}/call-tree", response_model=CallTreeResponse, tags=["analysis views"])
    def call_tree(
        analysis_id: str,
        request_id: FilterText = "",
        component_id: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> CallTreeResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return CallTreeResponse(
            **queries.call_tree_projection(
                request_id=request_id,
                component_id=component_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )

    @router.get(
        "/analyses/{analysis_id}/span-profile",
        response_model=SpanProfileResponse,
        tags=["analysis views"],
    )
    def span_profile(
        analysis_id: str,
        mode: Literal["request", "aggregate"] = "request",
        request_id: FilterText = "",
        component_id: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
        origin: FilterText = "",
        span_status: Annotated[str, Query(alias="status", max_length=256)] = "",
        max_nodes: SpanProfileNodeLimit = MAX_SPAN_PROFILE_NODES,
    ) -> SpanProfileResponse:
        if mode == "request" and not request_id:
            raise HTTPException(status_code=422, detail="request_id is required when mode=request")
        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            raise HTTPException(status_code=422, detail="start_ns must not be greater than end_ns")
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return queries.span_profile_projection(
            mode=mode,
            request_id=request_id,
            component_id=component_id,
            start_ns=start_ns,
            end_ns=end_ns,
            origin=origin,
            status=span_status,
            max_nodes=max_nodes,
        )

    @router.get(
        "/analyses/{analysis_id}/critical-path",
        response_model=CriticalPathResponse,
        tags=["analysis views"],
    )
    def critical_path(
        analysis_id: str,
        request_id: RequiredRequestId,
        component_id: FilterText = "",
        start_ns: int | None = None,
        end_ns: int | None = None,
        include_flows: bool = True,
        max_segments: CriticalPathSegmentLimit = DEFAULT_MAX_CRITICAL_PATH_SEGMENTS,
    ) -> CriticalPathResponse:
        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            raise HTTPException(status_code=422, detail="start_ns must not be greater than end_ns")
        queries = ResultQueries(_entry(manager, analysis_id).result)
        try:
            document = queries.critical_path_projection(
                request_id=request_id,
                component_id=component_id,
                start_ns=start_ns,
                end_ns=end_ns,
                include_flows=include_flows,
                max_segments=max_segments,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CriticalPathResponse(**document)

    @router.get("/analyses/{analysis_id}/graph", response_model=GraphResponse, tags=["analysis views"])
    def graph(
        analysis_id: str,
        request_id: FilterText = "",
        metric: Literal["p50", "p95", "p99", "minimum", "maximum", "mean"] = "p95",
        view: Literal["nodes", "components", "tracepoints"] = "components",
    ) -> GraphResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        return GraphResponse(**queries.graph_projection(request_id=request_id, metric=metric, view=view))

    @router.get("/analyses/{analysis_id}/warnings", response_model=WarningListResponse, tags=["analysis views"])
    def warnings(analysis_id: str) -> WarningListResponse:
        queries = ResultQueries(_entry(manager, analysis_id).result)
        total = len(queries.result.warnings)
        return WarningListResponse(items=queries.warnings(limit=100), total=total, limit=100, truncated=total > 100)

    app.include_router(router)
    if config.web_root is not None:
        web_root = config.web_root
        index_path = web_root / "index.html"
        asset_roots = {web_root, index_path.resolve().parent}

        @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
        def web_index() -> FileResponse:
            return FileResponse(index_path)

        @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
        def web_asset_or_route(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/") or path == "healthz" or path.startswith("healthz/"):
                raise HTTPException(status_code=404, detail="Not Found")
            asset_path = (web_root / path).resolve()
            if any(asset_path.is_relative_to(root) for root in asset_roots) and asset_path.is_file():
                return FileResponse(asset_path)
            if path.startswith("assets/") or Path(path).suffix:
                raise HTTPException(status_code=404, detail="Not Found")
            return FileResponse(index_path)

    return app
