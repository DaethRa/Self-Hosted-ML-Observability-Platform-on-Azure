import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/alerts":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                self.send_error(413)
                return

            payload = json.loads(self.rfile.read(length))

            for alert in payload.get("alerts", []):
                print(
                    "ALERT RECEIVED:",
                    json.dumps({
                        "status": alert.get("status"),
                        "alertname": alert.get("labels", {}).get("alertname"),
                        "severity": alert.get("labels", {}).get("severity"),
                        "feature": alert.get("labels", {}).get("feature"),
                        "summary": alert.get("annotations", {}).get("summary"),
                    }),
                    flush=True,
                )

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except (ValueError, json.JSONDecodeError) as exc:
            print("Invalid payload:", exc, flush=True)
            self.send_error(400)


print("Webhook receiver listening on port 8080", flush=True)
HTTPServer(("0.0.0.0", 8080), AlertHandler).serve_forever()
