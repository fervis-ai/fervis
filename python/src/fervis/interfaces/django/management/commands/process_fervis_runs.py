from __future__ import annotations

import signal
import socket
import time
import uuid

from django.core.management.base import BaseCommand
from django.db import OperationalError, ProgrammingError, close_old_connections

from fervis.interfaces.django.worker import process_run_batch


class Command(BaseCommand):
    help = "Process queued Fervis runs."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", default=False)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--sleep-seconds", type=float, default=1.0)
        parser.add_argument("--lease-seconds", type=int, default=300)
        parser.add_argument("--worker-id", default="")

    def handle(self, *args, **options):
        self._shutdown_requested = False
        once = bool(options.get("once"))
        batch_size = max(1, int(options.get("batch_size") or 1))
        sleep_seconds = max(0.1, float(options.get("sleep_seconds") or 1.0))
        lease_seconds = max(30, int(options.get("lease_seconds") or 300))
        worker_id = str(options.get("worker_id") or "").strip() or _worker_id()

        if not once:
            signal.signal(signal.SIGTERM, self._on_shutdown_signal)
            signal.signal(signal.SIGINT, self._on_shutdown_signal)

        self._wait_for_schema()
        self.stdout.write(
            self.style.SUCCESS(
                "Starting Fervis worker "
                f"(worker_id={worker_id}, batch_size={batch_size}, "
                f"sleep_seconds={sleep_seconds}, lease_seconds={lease_seconds})"
            )
        )

        last_heartbeat_at = 0.0
        while not self._shutdown_requested:
            close_old_connections()
            cycle = process_run_batch(
                worker_id=worker_id,
                batch_size=batch_size,
                lease_seconds=lease_seconds,
            )
            if cycle.claimed_count:
                self.stdout.write(
                    "Fervis worker cycle: "
                    f"claimed={cycle.claimed_count} "
                    f"completed={cycle.completed_count} "
                    f"failed={cycle.failed_count} "
                    f"queue={cycle.queue_counts}"
                )
            else:
                now_mono = time.monotonic()
                if now_mono - last_heartbeat_at >= 60:
                    last_heartbeat_at = now_mono
                    self.stdout.write(
                        f"Fervis worker heartbeat: queue={cycle.queue_counts}"
                    )

            if once:
                break
            time.sleep(sleep_seconds)

        self.stdout.write("Fervis worker stopped.")

    def _wait_for_schema(self) -> None:
        from fervis.run_work.queue.django.models import (
            RunWorkItem,
        )

        deadline = time.monotonic() + 60
        while True:
            try:
                RunWorkItem.objects.exists()
                return
            except (OperationalError, ProgrammingError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def _on_shutdown_signal(self, _signum, _frame):
        self._shutdown_requested = True


def _worker_id() -> str:
    return f"{socket.gethostname()}:{uuid.uuid4()}"
