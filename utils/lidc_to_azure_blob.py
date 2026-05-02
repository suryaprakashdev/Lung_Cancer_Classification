"""
LIDC-IDRI → Azure Blob Storage ingestion (IDC source, parallel)
---------------------------------------------------------------
Uses NCI Imaging Data Commons (IDC) as the data source.
Processes multiple series concurrently using a thread pool.
Nothing is written to local disk.

Prerequisites
-------------
    pip install azure-storage-blob requests rich python-dotenv idc-index
    Requires Python 3.10+

Setup
-----
Create a .env file at the project root:
    AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
    AZURE_CONTAINER_NAME=lidc-idri

Usage
-----
    python utils/lidc_to_azure_blob.py
"""

import os
import json
import time
import logging
import datetime
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    FileSizeColumn,
    TransferSpeedColumn,
    SpinnerColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceExistsError

from idc_index.index import IDCClient

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

COLLECTION          = "lidc_idri"
BLOB_CONN_STR       = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "YOUR_CONNECTION_STRING_HERE")
CONTAINER_NAME      = os.environ.get("AZURE_CONTAINER_NAME", "lidc-idri")
CHECKPOINT_FILE     = Path(__file__).parent / "progress_checkpoint.json"
LOG_FILE            = Path(__file__).parent.parent / "logs" / "lidc_ingest.log"

WORKERS             = 8     # concurrent series — tune up/down based on your bandwidth
MAX_RETRIES         = 5
RETRY_BACKOFF_BASE  = 10
CHUNK_SIZE          = 4 * 1024 * 1024   # 4 MB

console = Console()

# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger("lidc_ingest")
log.setLevel(logging.DEBUG)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
log.addHandler(file_handler)

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Thread-local requests session (one per thread for connection pooling) ──────

_thread_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ── Checkpoint ─────────────────────────────────────────────────────────────────

_checkpoint_lock = threading.Lock()

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(
            {"completed": list(completed), "count": len(completed),
             "updated_at": datetime.datetime.now().isoformat()},
            f, indent=2,
        )
    tmp.replace(CHECKPOINT_FILE)


# ── URL converter ──────────────────────────────────────────────────────────────

def to_https(url: str) -> str:
    if url.startswith("gs://"):
        return url.replace("gs://", "https://storage.googleapis.com/", 1)
    if url.startswith("s3://"):
        parts  = url[5:].split("/", 1)
        bucket = parts[0]
        key    = parts[1] if len(parts) > 1 else ""
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


# ── Upload one DICOM file ──────────────────────────────────────────────────────

def upload_file(container_client, url: str, blob_name: str) -> int:
    """
    Stream one DICOM file into Azure Blob.
    Returns bytes uploaded, 0 if blob already exists, -1 on permanent failure.
    """
    https_url = to_https(url)
    session   = get_session()

    for attempt in range(1, MAX_RETRIES + 1):
        bytes_uploaded = 0
        try:
            with session.get(https_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total_bytes  = int(resp.headers.get("Content-Length", 0)) or None
                blob_client: BlobClient = container_client.get_blob_client(blob_name)

                def chunks():
                    nonlocal bytes_uploaded
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            bytes_uploaded += len(chunk)
                            yield chunk

                blob_client.upload_blob(data=chunks(), overwrite=False, length=total_bytes)
            return bytes_uploaded

        except ResourceExistsError:
            return 0

        except Exception as exc:
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {blob_name}: {exc}. Retrying in {wait}s")
            time.sleep(wait)

    log.error(f"Giving up on {blob_name}")
    return -1


# ── Process one complete series (called from thread pool) ──────────────────────

def process_series(
    container_client,
    idc_client,
    series_uid: str,
    patient_id: str,
    modality: str,
    worker_progress: Progress,
    worker_task,
) -> tuple[bool, int]:
    """
    Download and upload all DICOM files for one series.
    Returns (success, bytes_uploaded).
    """
    try:
        file_urls = idc_client.get_series_file_URLs(
            seriesInstanceUID=series_uid,
            source_bucket_location="aws",
        )
    except Exception as exc:
        log.error(f"Could not get URLs for {series_uid}: {exc}")
        return False, 0

    total_files   = len(file_urls)
    series_bytes  = 0

    worker_progress.update(
        worker_task,
        description=f"[dim]{patient_id} | {modality} | {total_files} files[/dim]",
        total=total_files,
        completed=0,
    )

    for file_url in file_urls:
        filename  = file_url.split("/")[-1]
        blob_name = f"{patient_id}/{series_uid}/{filename}"

        result = upload_file(container_client, file_url, blob_name)

        if result == -1:
            log.error(f"Failed series {series_uid} on file {filename}")
            return False, series_bytes

        series_bytes += result
        worker_progress.advance(worker_task)

    log.info(f"✓  {patient_id} | {modality} | {series_uid} | {total_files} files | {series_bytes/1024/1024:.1f} MB")
    return True, series_bytes


# ── Utilities ──────────────────────────────────────────────────────────────────

def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold cyan]LIDC-IDRI → Azure Blob Storage Ingestion[/bold cyan]\n"
        f"Source     : [yellow]NCI Imaging Data Commons (AWS public)[/yellow]\n"
        f"Collection : [yellow]{COLLECTION}[/yellow]\n"
        f"Workers    : [yellow]{WORKERS} parallel series[/yellow]\n"
        f"Container  : [yellow]{CONTAINER_NAME}[/yellow]\n"
        f"Log file   : [yellow]{LOG_FILE}[/yellow]\n"
        f"Checkpoint : [yellow]{CHECKPOINT_FILE}[/yellow]",
        box=box.ROUNDED,
    ))

    # ── IDC index ─────────────────────────────────────────────────────────────
    console.print("\n[dim]Loading IDC index…[/dim]")
    idc_client = IDCClient()
    index_df   = idc_client.index
    series_df  = index_df[index_df["collection_id"] == COLLECTION][
        ["SeriesInstanceUID", "PatientID", "Modality"]
    ].drop_duplicates(subset="SeriesInstanceUID").reset_index(drop=True)

    total_series = len(series_df)
    console.print(f"[green]✓ Found {total_series} series (from local index)[/green]\n")
    log.info(f"Found {total_series} series in {COLLECTION}")

    # ── Azure ─────────────────────────────────────────────────────────────────
    console.print("[dim]Connecting to Azure Blob Storage…[/dim]")
    service_client   = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    container_client = service_client.get_container_client(CONTAINER_NAME)

    try:
        container_client.create_container()
        log.info(f"Created container '{CONTAINER_NAME}'")
    except ResourceExistsError:
        log.info(f"Container '{CONTAINER_NAME}' already exists")

    # ── Resume ────────────────────────────────────────────────────────────────
    completed = load_checkpoint()
    if completed:
        console.print(f"[yellow]Resuming — {len(completed)}/{total_series} already done.[/yellow]\n")
        log.info(f"Resuming: {len(completed)} series in checkpoint")

    pending = [
        row for row in series_df.itertuples()
        if row.SeriesInstanceUID not in completed
    ]

    # ── Shared counters (thread-safe) ─────────────────────────────────────────
    lock         = threading.Lock()
    ok_count     = len(completed)
    fail_count   = 0
    total_bytes  = 0
    session_start = time.time()

    # ── Progress ──────────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=35),
        TaskProgressColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]elapsed |[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]left[/dim]"),
        console=console,
        refresh_per_second=4,
    ) as overall_progress, Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        console=console,
        refresh_per_second=4,
    ) as worker_progress:

        overall_task = overall_progress.add_task(
            f"[cyan]Series {len(completed)}/{total_series}  ✓{ok_count}  ✗0  {format_bytes(0)} uploaded",
            total=total_series,
            completed=len(completed),
        )

        # One progress bar slot per worker
        worker_tasks = [
            worker_progress.add_task(f"[dim]Worker {i+1} idle[/dim]", total=1, completed=0)
            for i in range(WORKERS)
        ]

        # Map futures → worker slot index
        future_to_worker: dict = {}

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # Submit all pending series
            for i, row in enumerate(pending):
                worker_slot = i % WORKERS
                future = executor.submit(
                    process_series,
                    container_client,
                    idc_client,
                    row.SeriesInstanceUID,
                    row.PatientID,
                    row.Modality,
                    worker_progress,
                    worker_tasks[worker_slot],
                )
                future_to_worker[future] = worker_slot

            for future in as_completed(future_to_worker):
                success, series_bytes = future.result()

                # Find which series this was (reverse lookup)
                idx = list(future_to_worker.keys()).index(future)
                row = pending[idx]

                with lock:
                    if success:
                        ok_count    += 1
                        total_bytes += series_bytes
                        completed.add(row.SeriesInstanceUID)
                        save_checkpoint(completed)
                    else:
                        fail_count += 1

                    overall_progress.advance(overall_task)
                    overall_progress.update(
                        overall_task,
                        description=(
                            f"[cyan]Series {ok_count + fail_count}/{total_series}  "
                            f"[green]✓{ok_count}[/green]  "
                            f"[red]✗{fail_count}[/red]  "
                            f"[dim]{format_bytes(total_bytes)} uploaded[/dim]"
                        ),
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed   = time.time() - session_start
    elapsed_s = str(datetime.timedelta(seconds=int(elapsed)))

    table = Table(title="\nSession Summary", box=box.SIMPLE_HEAVY, show_header=False, min_width=44)
    table.add_column(style="dim", width=22)
    table.add_column(style="bold")

    table.add_row("Total series",      str(total_series))
    table.add_row("Uploaded this run", f"[green]{ok_count - len(load_checkpoint()) + (ok_count - len(load_checkpoint()))}[/green]")
    table.add_row("Failed",            f"[red]{fail_count}[/red]" if fail_count else "[green]0[/green]")
    table.add_row("Data uploaded",     format_bytes(total_bytes))
    table.add_row("Session duration",  elapsed_s)
    table.add_row("Log file",          str(LOG_FILE))
    table.add_row("Checkpoint file",   str(CHECKPOINT_FILE))

    console.print(table)

    if fail_count:
        console.print(f"\n[red]{fail_count} series failed.[/red] Re-run — they will be retried automatically.")
    else:
        console.print("\n[bold green]All series uploaded successfully.[/bold green]")

    log.info(
        f"Session complete | Uploaded: {ok_count} | Failed: {fail_count} | "
        f"Data: {format_bytes(total_bytes)} | Duration: {elapsed_s}"
    )


if __name__ == "__main__":
    main()