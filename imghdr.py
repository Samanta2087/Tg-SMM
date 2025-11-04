# Minimal imghdr shim for Python 3.13+ (stdlib imghdr removed)
# Required by python-telegram-bot library

def what(file, h=None):
    if h is None:
        if isinstance(file, (str, bytes)):
            try:
                with open(file, "rb") as f:
                    h = f.read(32)
            except Exception:
                return None
        else:
            try:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
            except Exception:
                return None
    if not h:
        return None
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    elif h[:2] == b"\xff\xd8":
        return "jpeg"
    elif h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    elif h[:2] == b"BM":
        return "bmp"
    elif h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "webp"
    elif h[:2] in (b"MM", b"II"):
        return "tiff"
    return None

def tests():
    return []
