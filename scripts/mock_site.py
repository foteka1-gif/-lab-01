"""Локальний тестовий HTTP-сервер для краулера: сторінки, затримки, robots.txt і збої."""

import argparse
import asyncio
from urllib.parse import parse_qs, urlsplit


class MockSiteServer:
    def __init__(self, total_pages: int = 260, latency: float = 0.05) -> None:
        self.total_pages = total_pages
        self.latency = latency
        self._flaky_hits: dict[str, int] = {}

    def _render_page(self, page_id: int) -> tuple[int, dict[str, str], bytes]:
        next_1 = (page_id + 1) % self.total_pages
        next_2 = (page_id + 7) % self.total_pages
        next_3 = (page_id * 3 + 1) % self.total_pages

        extra_links = ""
        if page_id == 0:
            extra_links = (
                '<a href="/private/secret">Таємна сторінка</a> '
                '<a href="/flaky-429">Ліміт 429</a> '
                '<a href="/flaky-503">Помилка 503</a> '
                '<a href="/missing-404">Немає 404</a>'
            )

        body = f"""<!doctype html>
<html>
<head><title>Сторінка #{page_id} — Тестова документація findex</title></head>
<body>
  <h1>Документ номер {page_id}</h1>
  <p>Це тестова сторінка про асинхронний краулер на Python, asyncio, TaskGroup та httpx.</p>
  <p>Ключові слова для пошуку: корутина, семафор, черга, повтори, сторінка_{page_id}.</p>
  <nav>
    <a href="/page/{next_1}?b=2&a=1#top">Далі {next_1}</a>
    <a href="/page/{next_2}#section">Посилання {next_2}</a>
    <a href="/page/{next_3}">Посилання {next_3}</a>
    {extra_links}
  </nav>
</body>
</html>
""".encode()
        return 200, {"Content-Type": "text/html; charset=utf-8"}, body

    async def handle_request(self, path: str) -> tuple[int, dict[str, str], bytes]:
        if self.latency > 0:
            await asyncio.sleep(self.latency)

        if path == "/robots.txt":
            txt = b"User-agent: *\nDisallow: /private/\n"
            return 200, {"Content-Type": "text/plain; charset=utf-8"}, txt

        if path.startswith("/private/"):
            return 200, {"Content-Type": "text/html; charset=utf-8"}, b"<h1>Private</h1>"

        if path == "/flaky-429":
            hits = self._flaky_hits.get(path, 0) + 1
            self._flaky_hits[path] = hits
            if hits == 1:
                return 429, {"Content-Type": "text/plain", "Retry-After": "0.15"}, b"Rate limited"
            body = b"<html><head><title>Recovered 429</title></head><body>Recovered after 429</body></html>"
            return 200, {"Content-Type": "text/html; charset=utf-8"}, body

        if path == "/flaky-503":
            hits = self._flaky_hits.get(path, 0) + 1
            self._flaky_hits[path] = hits
            if hits == 1:
                return 503, {"Content-Type": "text/plain"}, b"Service Unavailable"
            body = b"<html><head><title>Recovered 503</title></head><body>Recovered after 503</body></html>"
            return 200, {"Content-Type": "text/html; charset=utf-8"}, body

        if path == "/missing-404":
            return 404, {"Content-Type": "text/plain"}, b"Not Found"

        if path == "/" or path == "/page/0":
            return self._render_page(0)

        if path.startswith("/page/"):
            try:
                page_id = int(path.removeprefix("/page/"))
            except ValueError:
                return 404, {"Content-Type": "text/plain"}, b"Bad page id"
            if 0 <= page_id < self.total_pages:
                return self._render_page(page_id)

        return 404, {"Content-Type": "text/plain"}, b"Not Found"

    async def client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw_line = await reader.readline()
            if not raw_line:
                return
            parts = raw_line.decode("latin-1").strip().split(" ")
            raw_target = parts[1] if len(parts) >= 2 else "/"
            while True:
                header_line = await reader.readline()
                if not header_line or header_line == b"\r\n":
                    break

            split = urlsplit(raw_target)
            _ = parse_qs(split.query)
            status, headers, body = await self.handle_request(split.path)
            reason = {
                200: "OK",
                404: "Not Found",
                429: "Too Many Requests",
                503: "Service Unavailable",
            }.get(status, "OK")

            headers_out = {
                "Content-Length": str(len(body)),
                "Connection": "close",
                **headers,
            }
            head_lines = [f"HTTP/1.1 {status} {reason}"]
            for k, v in headers_out.items():
                head_lines.append(f"{k}: {v}")
            head_bytes = ("\r\n".join(head_lines) + "\r\n\r\n").encode("latin-1")

            writer.write(head_bytes + body)
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()


async def run_server(host: str, port: int, total_pages: int, latency: float) -> None:
    site = MockSiteServer(total_pages=total_pages, latency=latency)
    server = await asyncio.start_server(site.client_connected, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(
        f"MockSite запущено на http://{host}:{port} ({addrs}), сторінок={total_pages}, затримка={latency}с"
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальний тестовий сайт для краулера findex.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pages", type=int, default=260)
    parser.add_argument("--latency", type=float, default=0.05)
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.pages, args.latency))
    except KeyboardInterrupt:
        print("\nСервер зупинено.")


if __name__ == "__main__":
    main()
