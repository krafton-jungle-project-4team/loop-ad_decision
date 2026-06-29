from dataclasses import dataclass
from io import BytesIO
from textwrap import wrap

from app.contents.compose.png_canvas import PngCanvas


@dataclass(frozen=True)
class BannerRenderScene:
    copy: dict[str, str]


class PngExporterService:
    def export(self, composition_svg: str, scene: BannerRenderScene) -> bytes:
        try:
            import cairosvg  # type: ignore[import-not-found]
        except Exception:
            return self._export_with_builtin_renderer(scene)

        try:
            output = BytesIO()
            cairosvg.svg2png(
                bytestring=composition_svg.encode("utf-8"),
                write_to=output,
                output_width=1200,
                output_height=628,
            )
            return output.getvalue()
        except Exception as exc:
            raise RuntimeError("SVG to PNG export failed") from exc

    def _export_with_builtin_renderer(self, scene: BannerRenderScene) -> bytes:
        canvas = PngCanvas(1200, 628)
        # Repaint a close approximation so mock end-to-end works without native renderers.
        for y in range(628):
            for x in range(1200):
                left_weight = 1 - (x / 1199)
                vertical = y / 627
                red = int(238 + 14 * left_weight + 5 * vertical)
                green = int(247 - 18 * vertical)
                blue = int(231 - 33 * left_weight + 9 * vertical)
                canvas.set_pixel(x, y, (red, green, blue))

        copy = scene.copy
        canvas.fill_rect(70, 70, 176, 42, (229, 246, 232))
        canvas.draw_text(92, 84, copy["badge"][:13], 3, (15, 107, 79))
        canvas.draw_text(88, 135, copy["brand_name"][:18], 3, (15, 107, 79))

        headline_lines = wrap(copy["headline"], width=19)[:2]
        for index, line in enumerate(headline_lines):
            canvas.draw_text(88, 200 + index * 62, line, 7, (16, 37, 28))

        subcopy_lines = wrap(copy["subcopy"], width=38)[:2]
        for index, line in enumerate(subcopy_lines):
            canvas.draw_text(90, 335 + index * 32, line, 3, (49, 85, 72))

        canvas.fill_rect(88, 424, 290, 72, (15, 107, 79))
        canvas.draw_text(120, 449, copy["cta"][:20], 3, (255, 255, 255))

        canvas.fill_rect(805, 255, 282, 190, (240, 195, 106))
        canvas.fill_rect(835, 217, 222, 82, (248, 223, 158))
        canvas.fill_rect(855, 290, 70, 80, (91, 170, 114))
        canvas.fill_rect(935, 275, 70, 95, (244, 126, 83))
        canvas.fill_rect(1010, 290, 60, 80, (216, 231, 102))
        canvas.fill_rect(843, 313, 204, 64, (255, 248, 230))
        canvas.draw_text(875, 333, "FRESH", 5, (15, 107, 79))

        return canvas.to_png()
