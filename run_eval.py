"""Run PiCon evaluation against the local Jiyu Kim persona server.

This script intentionally does not bundle the upstream PiCon framework.
Prepare PiCon separately, then either:
1) install it into the current environment, or
2) set PICON_SOURCE_DIR to the upstream repository root.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("run_eval")

ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = ROOT / "rag_persona_server.py"
RESULTS_DIR = ROOT / "outputs" / "jiyu_kim"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DEFAULT_PORT = 8001


def ensure_picon_importable() -> None:
    try:
        importlib.import_module("picon")
        return
    except ModuleNotFoundError:
        pass

    source_dir = os.environ.get("PICON_SOURCE_DIR")
    if source_dir:
        source_path = Path(source_dir).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"PICON_SOURCE_DIR does not exist: {source_path}")
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        try:
            importlib.import_module("picon")
            return
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Could not import 'picon' even after adding PICON_SOURCE_DIR. "
                "Point PICON_SOURCE_DIR to the upstream PiCon repo root."
            ) from exc

    raise SystemExit(
        "Could not import 'picon'. Install the upstream PiCon package first, "
        "or set PICON_SOURCE_DIR=/path/to/picon-repo."
    )


def stream_logs(proc: subprocess.Popen, tag: str) -> None:
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            log.info("[%s] %s", tag, text)


def wait_for_server(url: str, timeout: int = 120) -> bool:
    log.info("Waiting for server at %s ...", url)
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.post(
                f"{url}/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                timeout=10,
            )
            if response.status_code == 200:
                log.info("Server ready! (%.0fs)", time.time() - start)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--eval_factors",
        nargs="+",
        default=["internal", "external", "intra", "inter"],
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")

    ensure_picon_importable()
    import picon

    procs: list[subprocess.Popen] = []

    def cleanup() -> None:
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))

    try:
        log.info("=" * 60)
        log.info("[1/3] Starting rag_persona_server (Jiyu Kim)...")
        log.info("      Model: %s | Port: %d", args.model, args.port)
        log.info("=" * 60)

        cmd = [
            sys.executable,
            str(SERVER_SCRIPT),
            "--port",
            str(args.port),
            "--model",
            args.model,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        procs.append(proc)
        threading.Thread(target=stream_logs, args=(proc, "SERVER"), daemon=True).start()

        if not wait_for_server(f"http://localhost:{args.port}/v1"):
            log.error("Server failed to start.")
            cleanup()
            return 1

        log.info("=" * 60)
        log.info("[2/3] Quick sanity check...")
        log.info("=" * 60)
        sanity = requests.post(
            f"http://localhost:{args.port}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "Can you tell me your year of birth?"}
                ],
            },
            timeout=30,
        )
        sanity.raise_for_status()
        answer = sanity.json()["choices"][0]["message"]["content"]
        log.info("Test response: %s", answer)

        log.info("=" * 60)
        log.info("[3/3] Running PiCon evaluation...")
        log.info("      Turns: %d | Sessions: %d", args.turns, args.sessions)
        log.info("      Factors: %s", args.eval_factors)
        log.info("=" * 60)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result = picon.run(
            persona="",
            api_base=f"http://localhost:{args.port}/v1",
            name="Jiyu Kim",
            num_turns=args.turns,
            num_sessions=args.sessions,
            do_eval=True,
            eval_factors=args.eval_factors,
            output_dir=str(RESULTS_DIR),
            questioner_model="gpt-4o-mini",
            extractor_model="gpt-4o-mini",
            web_search_model="gpt-4o-mini",
            evaluator_model="gpt-4o-mini",
        )

        if not result.result_path:
            log.error("Interview failed — no result file saved. Check server logs.")
            return 1

        log.info("Result path: %s", result.result_path)
        log.info("Success: %s", result.success)

        with open(result.result_path, encoding="utf-8") as fp:
            full_result = json.load(fp)

        stability_scores = picon.evaluate(
            result.result_path,
            eval_factors=["intra", "inter"],
            evaluator_model="gpt-4o-mini",
        )

        summary: dict[str, object] = {"stability": stability_scores}
        session_keys = sorted(k for k in full_result if k.startswith("session_"))
        for session_key in [k for k in session_keys if k == "session_1"]:
            tmp_path = RESULTS_DIR / f"{session_key}_temp.json"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump({session_key: full_result[session_key]}, fp)
            session_scores = picon.evaluate(
                str(tmp_path),
                eval_factors=["internal", "external"],
                evaluator_model="gpt-4o-mini",
            )
            summary[session_key] = session_scores
            tmp_path.unlink(missing_ok=True)

        with open(SUMMARY_PATH, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2, default=str)
        log.info("Summary saved to %s", SUMMARY_PATH)
        return 0

    except Exception as exc:
        log.exception("Evaluation failed: %s", exc)
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
