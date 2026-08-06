"""
Serves the generated LLM Delusions Evaluation report on a local web server
and automatically opens the browser.
"""

import argparse
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


def serve_report(report_dir: str, port: int) -> None:
    """Starts a local web server to view the report dashboard."""
    if not os.path.isdir(report_dir):
        print(f"Error: '{report_dir}' missing. Generate the report first.")
        return

    os.chdir(report_dir)
    server_address = ("", port)

    try:
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    except OSError as e:
        print(f"Error starting server on port {port}: {e}")
        return

    url = f"http://localhost:{port}/index.html"
    print(f"\nServing dashboard from '{report_dir}' at {url}")
    print("Press Ctrl+C to stop the server.")

    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main() -> None:
    """Main entry point for serving the evaluation report."""
    parser = argparse.ArgumentParser(
        description="Serve the evaluation report dashboard locally."
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="report",
        help="Path to the generated report directory (default: report/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for the local web server (default: 8000).",
    )
    args = parser.parse_args()

    serve_report(args.report_dir, args.port)


if __name__ == "__main__":
    main()
