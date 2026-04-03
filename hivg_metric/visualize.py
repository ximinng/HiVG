"""
HiVG Evaluation Visualizer

Generate HTML report from evaluation results.

Usage:
    python hivg_metric/visualize.py \
        --input results.jsonl \
        --output report.html \
        --title "HiVG Evaluation Report"
"""

import html
import json
import os
import re
from typing import Optional

import fire


def _extract_image_from_prompt(prompt: str) -> Optional[str]:
    """Extract image path from prompt if present."""
    # Common patterns for image references in prompts
    # Pattern 1: <image>path</image>
    match = re.search(r"<image>(.*?)</image>", prompt)
    if match:
        return match.group(1)

    # Pattern 2: image path in the prompt text
    # Look for common image extensions
    match = re.search(r'([^\s<>"]+\.(?:png|jpg|jpeg|gif|webp|svg))', prompt, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _clean_prompt_for_display(prompt: str) -> str:
    """Clean prompt for display, removing image tags and paths."""
    # Remove <image> tags
    cleaned = re.sub(r"<image>.*?</image>", "[IMAGE]", prompt)
    # Remove standalone image paths
    cleaned = re.sub(r'([^\s<>"]+\.(?:png|jpg|jpeg|gif|webp|svg))', "[IMAGE]", cleaned, flags=re.IGNORECASE)
    # Remove image placeholder tokens
    cleaned = re.sub(r"<\|image_pad\|>+", "[IMAGE]", cleaned)
    cleaned = re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "[IMAGE]", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _is_img2svg_task(prompt: str) -> bool:
    """Detect if this is an img2svg task based on prompt."""
    indicators = [
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "[I2ST]",
        "image to svg",
        "convert this image",
        "vectorize",
    ]
    prompt_lower = prompt.lower()
    return any(ind.lower() in prompt_lower for ind in indicators)


def _render_pagination_controls(current_page: int, total_pages: int) -> str:
    """Render pagination controls HTML."""
    first_disabled = "disabled" if current_page <= 1 else ""
    last_disabled = "disabled" if current_page >= total_pages else ""
    return (
        f'<button class="page-btn" onclick="goToPage(1)" {first_disabled}>First</button>'
        f'<button class="page-btn" onclick="goToPage({current_page - 1})" {first_disabled}>Prev</button>'
        f'<span class="page-info">Page {current_page} / {total_pages}</span>'
        f'<button class="page-btn" onclick="goToPage({current_page + 1})" {last_disabled}>Next</button>'
        f'<button class="page-btn" onclick="goToPage({total_pages})" {last_disabled}>Last</button>'
    )


def _generate_html_report(
    results: list,
    title: str = "HiVG Evaluation Report",
    image_base_dir: Optional[str] = None,
    page_size: int = 2000,
) -> str:
    """Generate HTML report from evaluation results."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = (success_count / total * 100) if total > 0 else 0
    pagination_placeholder = "__HIVG_PAGINATION__"
    page_chars_budget = 8 * 1024 * 1024
    page_cards = [[]]
    current_page_chars = 0

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }}
        .filters {{
            display: flex;
            gap: 10px;
            margin-left: auto;
        }}
        .filter-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: white;
            color: #667eea;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }}
        .card.hidden {{
            display: none;
        }}
        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            flex-wrap: wrap;
        }}
        .page-btn {{
            background: white;
            border: 1px solid #ddd;
            color: #667eea;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .page-btn:hover {{
            background: #f0f0f0;
        }}
        .page-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        .page-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .page-info {{
            color: #666;
            font-size: 14px;
        }}
        .card-header {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card-index {{
            font-weight: 600;
            color: #666;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .card-body {{
            padding: 20px;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }}
        .comparison-item {{
            text-align: center;
        }}
        .comparison-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .svg-container {{
            background: #fafafa;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 10px;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .svg-container svg {{
            max-width: 100%;
            max-height: 300px;
            width: auto;
            height: auto;
        }}
        .svg-container img {{
            max-width: 100%;
            max-height: 300px;
            object-fit: contain;
        }}
        .prompt-container {{
            background: #f8f9fa;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 15px;
            min-height: 200px;
            font-size: 13px;
            color: #555;
            overflow: auto;
            max-height: 300px;
            text-align: left;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .empty-state {{
            color: #999;
            font-style: italic;
        }}
        .download-btn {{
            display: inline-block;
            margin-top: 6px;
            padding: 4px 12px;
            font-size: 12px;
            color: #667eea;
            background: #f0f0f0;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .download-btn:hover {{
            background: #667eea;
            color: white;
        }}
        .tokens-section {{
            margin-top: 15px;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }}
        .tokens-toggle {{
            background: #f0f0f0;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            color: #666;
            transition: all 0.2s;
        }}
        .tokens-toggle:hover {{
            background: #e0e0e0;
        }}
        .tokens-content {{
            display: none;
            margin-top: 15px;
        }}
        .tokens-content.show {{
            display: block;
        }}
        .tokens-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        .tokens-box {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 11px;
            overflow: auto;
            max-height: 200px;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .tokens-box-label {{
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        @media (max-width: 900px) {{
            .comparison {{
                grid-template-columns: 1fr;
            }}
            .tokens-grid {{
                grid-template-columns: 1fr;
            }}
            .filters {{
                margin-left: 0;
                margin-top: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html.escape(title)}</h1>
        <div class="stats">
            <span class="stat-item">Total: {total}</span>
            <span class="stat-item">Success: {success_count} ({success_rate:.1f}%)</span>
            <span class="stat-item">Failed: {total - success_count}</span>
            <div class="filters">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="success">✓ Success</button>
                <button class="filter-btn" data-filter="failed">✗ Failed</button>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="pagination" id="pagination-top">{pagination_placeholder}</div>
        <div id="cards-container">
"""

    # Generate cards for each result
    for i, result in enumerate(results):
        idx = result.get("index", 0)
        prompt = result.get("prompt", "")
        raw_output = result.get("raw_output", "")
        svg = result.get("svg", "")
        label = result.get("label", "")
        label_svg = result.get("label_svg", "")
        success = result.get("success", False)

        status_class = "status-success" if success else "status-failed"
        status_text = "Success" if success else "Failed"
        filter_class = "success" if success else "failed"

        # Detect task type
        is_img2svg = _is_img2svg_task(prompt)

        # Clean prompt for display
        display_prompt = _clean_prompt_for_display(prompt)

        # Extract image path for img2svg
        image_path = _extract_image_from_prompt(prompt) if is_img2svg else None
        if image_path and image_base_dir:
            image_path = os.path.join(image_base_dir, image_path)

        # Input column content
        if is_img2svg and image_path and os.path.exists(image_path):
            # Read image and embed as base64 for portability
            import base64
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/png")
            input_html = f'<img src="data:{mime_type};base64,{img_data}" alt="Input Image">'
        else:
            # Show prompt text
            empty_prompt_html = '<span class="empty-state">No prompt</span>'
            prompt_content = html.escape(display_prompt) if display_prompt else empty_prompt_html
            input_html = f'<div class="prompt-container">{prompt_content}</div>'

        # SVG content (prediction)
        if svg:
            pred_svg_html = svg  # SVG is already valid XML
        else:
            pred_svg_html = '<span class="empty-state">No SVG generated</span>'

        # SVG content (ground truth)
        if label_svg:
            gt_svg_html = label_svg
        else:
            gt_svg_html = '<span class="empty-state">No ground truth</span>'

        card_html = f"""
        <div class="card" data-status="{filter_class}">
            <div class="card-header">
                <span class="card-index">#{idx}</span>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
            <div class="card-body">
                <div class="comparison">
                    <div class="comparison-item">
                        <div class="comparison-label">Input {'(Image)' if is_img2svg else '(Text)'}</div>
                        <div class="svg-container">
                            {input_html}
                        </div>
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-label">Prediction</div>
                        <div class="svg-container">
                            {pred_svg_html}
                        </div>
                        {f'<button class="download-btn" onclick="downloadSvg(this, {idx}, \'pred\')">Download SVG</button>' if svg else ''}
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-label">Ground Truth</div>
                        <div class="svg-container">
                            {gt_svg_html}
                        </div>
                        {f'<button class="download-btn" onclick="downloadSvg(this, {idx}, \'gt\')">Download SVG</button>' if label_svg else ''}
                    </div>
                </div>
                <div class="tokens-section">
                    <button class="tokens-toggle" onclick="toggleTokens(this)">Show Tokens ▼</button>
                    <div class="tokens-content">
                        <div class="tokens-grid">
                            <div>
                                <div class="tokens-box-label">Prediction Tokens (raw_output)</div>
                                <div class="tokens-box">{html.escape(raw_output) if raw_output else "(empty)"}</div>
                            </div>
                            <div>
                                <div class="tokens-box-label">Ground Truth Tokens (label)</div>
                                <div class="tokens-box">{html.escape(label) if label else "(empty)"}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""
        current_page = page_cards[-1]
        should_split_by_count = len(current_page) >= page_size
        should_split_by_chars = (current_page_chars + len(card_html) > page_chars_budget) and len(current_page) > 0
        if should_split_by_count or should_split_by_chars:
            page_cards.append([])
            current_page = page_cards[-1]
            current_page_chars = 0

        current_page.append(card_html)
        current_page_chars += len(card_html)

    no_samples_html = """
        <div class="card">
            <div class="card-body">
                <span class="empty-state">No samples found</span>
            </div>
        </div>
"""
    if total == 0:
        page_cards = [[no_samples_html]]

    page_html_list = ["".join(cards) for cards in page_cards]
    total_pages = len(page_html_list)
    initial_pagination_html = _render_pagination_controls(current_page=1, total_pages=total_pages)
    html_content = html_content.replace(pagination_placeholder, initial_pagination_html)

    page_data_blocks = ""
    for page_idx, page_html in enumerate(page_html_list, start=1):
        # Prevent accidental script termination in page payload.
        safe_page_html = re.sub(r"</script", r"<\\/script", page_html, flags=re.IGNORECASE)
        page_data_blocks += (
            f'\n    <script type="text/plain" class="page-data" data-page="{page_idx}">'
            f"{safe_page_html}</script>"
        )

    # Close HTML
    html_content += f"""
        </div>
        <div class="pagination" id="pagination-bottom">{initial_pagination_html}</div>
    </div>
    {page_data_blocks}
    <script>
        const TOTAL_PAGES = {total_pages};
        let currentPage = 1;
        let currentFilter = 'all';

        function getPageHtml(page) {{
            const selector = 'script.page-data[data-page=\"' + page + '\"]';
            const dataNode = document.querySelector(selector);
            return dataNode ? dataNode.textContent : '';
        }}

        // Initialize pagination
        function initPagination() {{
            renderCurrentPage();
            renderPagination();
        }}

        // Render pagination controls
        function renderPagination() {{
            const firstDisabled = currentPage <= 1 ? 'disabled' : '';
            const lastDisabled = currentPage >= TOTAL_PAGES ? 'disabled' : '';
            const html =
                '<button class="page-btn" onclick="goToPage(1)" ' + firstDisabled + '>First</button>' +
                '<button class="page-btn" onclick="goToPage(' + (currentPage - 1) + ')" ' + firstDisabled + '>Prev</button>' +
                '<span class="page-info">Page ' + currentPage + ' / ' + TOTAL_PAGES + '</span>' +
                '<button class="page-btn" onclick="goToPage(' + (currentPage + 1) + ')" ' + lastDisabled + '>Next</button>' +
                '<button class="page-btn" onclick="goToPage(' + TOTAL_PAGES + ')" ' + lastDisabled + '>Last</button>';
            document.getElementById('pagination-top').innerHTML = html;
            document.getElementById('pagination-bottom').innerHTML = html;
        }}

        function applyFilterOnCurrentPage() {{
            document.querySelectorAll('#cards-container .card').forEach(card => {{
                const status = card.dataset.status;
                const matchFilter = currentFilter === 'all' || status === currentFilter;
                card.classList.toggle('hidden', !matchFilter);
            }});
        }}

        function renderCurrentPage() {{
            const container = document.getElementById('cards-container');
            if (!container) return;
            container.innerHTML = getPageHtml(currentPage);
            applyFilterOnCurrentPage();
        }}

        // Show specific page
        function showPage(page) {{
            if (page < 1 || page > TOTAL_PAGES) return;
            currentPage = page;
            renderCurrentPage();
            renderPagination();
            window.scrollTo(0, 0);
        }}

        // Go to page
        function goToPage(page) {{
            showPage(page);
        }}

        // Filter functionality
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                const filter = this.dataset.filter;
                currentFilter = filter;

                // Update active state
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                // Reset to page 1 and apply filter
                showPage(1);
            }});
        }});

        // Download SVG
        function downloadSvg(btn, idx, type) {{
            const container = btn.previousElementSibling;
            const svg = container.querySelector('svg');
            if (!svg) return;
            const blob = new Blob([svg.outerHTML], {{type: 'image/svg+xml'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `sample_${{idx}}_${{type}}.svg`;
            a.click();
            URL.revokeObjectURL(a.href);
        }}

        // Toggle tokens
        function toggleTokens(btn) {{
            const content = btn.nextElementSibling;
            const isShown = content.classList.toggle('show');
            btn.textContent = isShown ? 'Hide Tokens ▲' : 'Show Tokens ▼';
        }}

        // Initialize on load
        initPagination();
    </script>
</body>
</html>
"""
    return html_content


def visualize(
    input: str,
    output: str = "hivg_eval_report.html",
    title: str = "HiVG Evaluation Report",
    image_base_dir: Optional[str] = None,
    page_size: int = 2000,
):
    """
    Generate HTML visualization report from JSONL results.

    Args:
        input: Path to input JSONL file (output from hivg_metric.eval)
        output: Path to output HTML file
        title: Report title
        image_base_dir: Base directory for resolving image paths (for img2svg)
        page_size: Number of samples per page (default: 2000)
    """
    # Load results
    results = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"[Visualize] Loaded {len(results)} results from {input}")

    # Generate HTML
    html_content = _generate_html_report(
        results=results,
        title=title,
        image_base_dir=image_base_dir,
        page_size=page_size,
    )

    # Write output
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Visualize] Report saved to {output}")
    print(f"[Visualize] Open in browser: file://{os.path.abspath(output)}")


if __name__ == "__main__":
    fire.Fire(visualize)
