"""把本机正在运行的服务映射成公网 https 链接（免注册、免路由器设置）。

原理：用系统自带 ssh 连到 pinggy.io 的反向隧道端点（走 443 端口，能穿过
大多数只放行 https 的网络），由对方给你一个 https 域名，任何设备——手机流量、
异地电脑——打开即可访问。

用法：
    python scripts/share_public.py                # 默认转发 127.0.0.1:8000
    python scripts/share_public.py --port 8011
    python scripts/share_public.py --service localhost.run   # 备用通道

注意：
    · 免费隧道约 60 分钟后过期，过期后重新运行本脚本即可换一个新链接；
    · 链接是公开的，知道地址的人都能上传图片并触发检测，用完按 Ctrl+C 关闭；
    · 本脚本只建隧道，不启动服务——请先用 start_server.bat 起服务。
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

URL_PATTERN = re.compile(r"https://[A-Za-z0-9._-]+(?:pinggy-free\.link|pinggy\.net|lhr\.life)")

SERVICES = {
    # name: (host, port, 额外 ssh 参数)
    "pinggy": ("a.pinggy.io", 443, ["-p", "443", "-R0:localhost:{port}"]),
    "localhost.run": ("nokey@localhost.run", 22, ["-R", "80:localhost:{port}"]),
}


def local_service_alive(port: int) -> bool:
    """本机服务是否在监听（避免隧道建好了却指向空气）。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def print_banner(url: str, port: int) -> None:
    """大字打印链接，并尽量画一个扫码用的二维码。"""
    print("\n" + "=" * 68)
    print("  公网链接已就绪 —— 任何设备（含手机流量）打开下面地址：")
    print(f"\n    {url}\n")
    print("  · 手机可直接扫码（二维码在下方）")
    print("  · 电脑直接把地址粘到浏览器")
    print("  · 关闭方式：在本窗口按 Ctrl+C（链接立即失效）")
    print("=" * 68)
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        print(f"（未安装 qrcode，跳过二维码；直接手动输入 {url} 也可）")
    print(f"[提示] 这是本机 {port} 端口的外网映射，本机服务请勿关闭。\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="把本机服务映射成公网链接")
    parser.add_argument("--port", type=int, default=8000, help="本机服务端口")
    parser.add_argument("--service", default="pinggy", choices=sorted(SERVICES), help="隧道服务")
    arguments = parser.parse_args()

    if not local_service_alive(arguments.port):
        print(f"[错误] 本机 {arguments.port} 端口没有服务在跑。")
        print("       请先双击 start_server.bat（或 start_server_lan.bat）启动服务，再运行本脚本。")
        return 1

    host, ssh_port, extra = SERVICES[arguments.service]
    # 已知主机文件必须落到临时目录：传 "NUL" 会被 ssh 当成相对路径，
    # 在当前目录建出一个名叫 NUL 的垃圾文件（Windows 下还不好删）
    known_hosts = os.path.join(tempfile.gettempdir(), "seatproject_known_hosts")
    command = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=" + known_hosts,
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-T",
        "-p", str(ssh_port),
    ] + [part.format(port=arguments.port) for part in extra] + [host]

    print(f"[1/2] 正在连接 {arguments.service} 隧道服务 …")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    found_url = None
    deadline = time.time() + 45
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        print("      " + line.rstrip())
        match = URL_PATTERN.search(line)
        if match:
            found_url = match.group(0)
            break

    if not found_url:
        print("\n[错误] 45 秒内没拿到公网地址，可能被网络拦了。")
        print("       改用备用通道：python scripts/share_public.py --service localhost.run")
        process.terminate()
        return 1

    print_banner(found_url, arguments.port)

    # 保持隧道存活：持续消费输出，Ctrl+C 时一并关掉 ssh
    try:
        for line in process.stdout:
            text = line.strip()
            if text and "Permission denied" not in text:
                print("      " + text)
    except KeyboardInterrupt:
        print("\n[关闭] 正在断开隧道 …")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    print("[已断开] 公网链接失效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
