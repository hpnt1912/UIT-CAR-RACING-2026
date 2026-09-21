"""
live_view.py
Máy chủ web nhỏ, nhúng vào main.py, để xem TRỰC TIẾP (gần real-time) ảnh
camera + mask segmentation khi xe đang chạy, mà không cần GUI trong
container (không dùng cv2.imshow).

Cách hoạt động: main.py liên tục gọi viewer.update(anh_ghep) với ảnh mới
nhất; trình duyệt trên máy host mở http://localhost:<port>, trang web tự
động tải lại ảnh mới mỗi ~150ms, tạo cảm giác xem video trực tiếp.

Trong VS Code (Dev Container), cổng này thường tự động xuất hiện ở tab
"Ports" phía dưới - bấm vào đó (hoặc icon quả địa cầu) để mở bằng trình
duyệt máy host.
"""

import http.server
import socketserver
import threading

import cv2


class LiveViewer:
    def __init__(self, port: int = 8080, jpeg_quality: int = 70):
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._frame_bytes = None
        self._server = None
        self._thread = None

    def update(self, image_bgr):
        """Gọi hàm này mỗi khi có ảnh mới muốn hiển thị (ảnh BGR, dùng cv2)."""
        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if ok:
            with self._lock:
                self._frame_bytes = buf.tobytes()

    def _make_handler(self):
        viewer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # tắt log request ồn ào ra terminal

            def do_GET(self):
                if self.path.startswith("/frame.jpg"):
                    with viewer._lock:
                        data = viewer._frame_bytes
                    if data is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>UCR 2026 - Live View</title>
<style>
  body { background:#111; margin:0; display:flex; align-items:center;
         justify-content:center; height:100vh; }
  img { max-width:100%; max-height:100%; }
</style></head>
<body>
  <img id="f" src="/frame.jpg" />
  <script>
    setInterval(() => {
      document.getElementById('f').src = '/frame.jpg?' + Date.now();
    }, 150);
  </script>
</body></html>"""
                    encoded = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

        return Handler

    def start(self):
        handler = self._make_handler()
        self._server = socketserver.ThreadingTCPServer(("0.0.0.0", self.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[LiveView] Xem trực tiếp tại http://localhost:{self.port} "
              f"(hoặc mở qua tab 'Ports' trong VS Code nếu đang chạy trong container)")

    def stop(self):
        if self._server:
            self._server.shutdown()