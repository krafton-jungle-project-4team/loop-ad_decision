import base64
from html import escape
from textwrap import wrap


class SvgComposerService:
    def compose(
        self,
        *,
        background_png: bytes,
        copy: dict[str, str],
        assets: dict[str, dict[str, str]],
    ) -> str:
        encoded_background = base64.b64encode(background_png).decode("ascii")
        headline_lines = wrap(copy["headline"], width=24)[:2]
        subcopy_lines = wrap(copy["subcopy"], width=48)[:2]
        brand = escape(copy.get("brand_name") or assets["brand"]["brand_name"])

        headline_svg = "\n".join(
            f'<text x="88" y="{210 + index * 62}" class="headline">{escape(line)}</text>'
            for index, line in enumerate(headline_lines)
        )
        subcopy_svg = "\n".join(
            f'<text x="90" y="{335 + index * 34}" class="subcopy">{escape(line)}</text>'
            for index, line in enumerate(subcopy_lines)
        )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="628" viewBox="0 0 1200 628">
  <defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="16" stdDeviation="18" flood-color="#0b2d21" flood-opacity="0.18"/>
    </filter>
    <style>
      .brand {{ font: 700 26px Inter, Arial, sans-serif; fill: #0F6B4F; }}
      .badge {{ font: 700 24px Inter, Arial, sans-serif; fill: #0F6B4F; letter-spacing: 2px; }}
      .headline {{ font: 800 56px Inter, Arial, sans-serif; fill: #10251C; }}
      .subcopy {{ font: 500 25px Inter, Arial, sans-serif; fill: #315548; }}
      .cta {{ font: 800 24px Inter, Arial, sans-serif; fill: #FFFFFF; }}
    </style>
  </defs>
  <image href="data:image/png;base64,{encoded_background}" width="1200" height="628" preserveAspectRatio="xMidYMid slice"/>
  <rect x="70" y="70" width="176" height="42" rx="21" fill="#E5F6E8"/>
  <text x="92" y="99" class="badge">{escape(copy["badge"])}</text>
  <text x="88" y="146" class="brand">{brand}</text>
  {headline_svg}
  {subcopy_svg}
  <rect x="88" y="424" width="290" height="72" rx="36" fill="#0F6B4F"/>
  <text x="120" y="470" class="cta">{escape(copy["cta"])}</text>
  <g filter="url(#softShadow)" transform="translate(805 125)">
    <rect x="0" y="130" width="282" height="190" rx="34" fill="#F0C36A"/>
    <rect x="30" y="92" width="222" height="82" rx="24" fill="#F8DF9E"/>
    <circle cx="74" cy="116" r="45" fill="#5BAA72"/>
    <circle cx="145" cy="105" r="38" fill="#F47E53"/>
    <circle cx="212" cy="122" r="42" fill="#D8E766"/>
    <rect x="38" y="188" width="204" height="64" rx="22" fill="#FFF8E6"/>
    <text x="70" y="230" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="#0F6B4F">FRESH</text>
  </g>
</svg>"""
