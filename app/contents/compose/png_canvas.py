import struct
import zlib


Color = tuple[int, int, int]


def clamp_color(color: Color) -> Color:
    red, green, blue = color
    return (
        max(0, min(255, red)),
        max(0, min(255, green)),
        max(0, min(255, blue)),
    )


def write_png_rgb(width: int, height: int, pixels: bytearray) -> bytes:
    raw_rows = bytearray()
    row_length = width * 3
    for y in range(height):
        raw_rows.append(0)
        start = y * row_length
        raw_rows.extend(pixels[start : start + row_length])

    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT",
        zlib.compress(bytes(raw_rows), level=6),
    ) + chunk(b"IEND", b"")


class PngCanvas:
    def __init__(self, width: int, height: int, background: Color = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * width * height)

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset : offset + 3] = bytes(clamp_color(color))

    def fill_rect(self, x: int, y: int, width: int, height: int, color: Color) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        clamped_color = bytes(clamp_color(color))
        for row in range(y0, y1):
            offset = (row * self.width + x0) * 3
            self.pixels[offset : offset + (x1 - x0) * 3] = clamped_color * (x1 - x0)

    def draw_text(self, x: int, y: int, text: str, scale: int, color: Color) -> None:
        cursor_x = x
        for char in text.upper():
            if char == " ":
                cursor_x += 4 * scale
                continue
            glyph = FONT_5X7.get(char, FONT_5X7.get("?"))
            if glyph is None:
                cursor_x += 4 * scale
                continue
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(
                            cursor_x + column_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor_x += 6 * scale

    def to_png(self) -> bytes:
        return write_png_rgb(self.width, self.height, self.pixels)


def create_mock_background_png(width: int, height: int) -> bytes:
    canvas = PngCanvas(width, height)
    for y in range(height):
        for x in range(width):
            left_weight = 1 - (x / max(width - 1, 1))
            vertical = y / max(height - 1, 1)
            red = int(238 + 14 * left_weight + 5 * vertical)
            green = int(247 - 18 * vertical)
            blue = int(231 - 33 * left_weight + 9 * vertical)
            canvas.set_pixel(x, y, (red, green, blue))

    canvas.fill_rect(width - 320, 120, 230, 250, (241, 250, 233))
    canvas.fill_rect(width - 285, 155, 190, 30, (203, 230, 189))
    canvas.fill_rect(width - 250, 225, 65, 88, (250, 139, 92))
    canvas.fill_rect(width - 175, 210, 72, 110, (57, 157, 91))
    canvas.fill_rect(width - 350, 365, 285, 55, (255, 184, 77))
    return canvas.to_png()


FONT_5X7: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["11111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
}
