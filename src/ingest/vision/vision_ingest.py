import os
import time
import json
import threading
import http.server
import socketserver
import asyncio
from pathlib import Path
from typing import List, Dict
from playwright.async_api import async_playwright, Page

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/ltdb")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../../data/ltdb_vision")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
PORT = 8082 
SERVER_URL = f"http://localhost:{PORT}"
CONCURRENCY = 5

TARGET_SELECTORS = [
    {"type": "leipzig", "selector": "div.leipzig"},
    {"type": "phonology_table", "selector": "table.ltdb"},
    {"type": "example_box", "selector": "div.ltdb_eg"},
    {"type": "general_table", "selector": "table:not(.ltdb)"},
    {"type": "embedded_image", "selector": "img"}
]

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

def start_server(port=PORT):
    os.chdir(DATA_DIR)
    handler = QuietHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()

def get_html_files(root_dir: str) -> List[str]:
    html_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                html_files.append(rel_path)
    return html_files

async def capture_elements(page: Page, rel_path: str, metadata_list: List[Dict]):
    file_stem = Path(rel_path).stem
    rel_dir = os.path.dirname(rel_path)
    save_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    captured_count = 0
    
    for target in TARGET_SELECTORS:
        stype = target["type"]
        selector = target["selector"]
        
        elements = await page.locator(selector).all()
        
        for i, el in enumerate(elements):
            if not await el.is_visible():
                continue
                
            box = await el.bounding_box()
            if not box or box['width'] < 50 or box['height'] < 50:
                continue

            image_filename = f"{file_stem}_{stype}_{i}.png"
            image_path = os.path.join(save_dir, image_filename)
            rel_image_path = os.path.join("data", "ltdb_vision", rel_dir, image_filename)

            try:
                await el.screenshot(path=image_path)
                context = (await el.inner_text())[:200].replace('\n', ' ')

                meta = {
                    "image_id": f"{file_stem}_{stype}_{i}",
                    "parent_filename": os.path.basename(rel_path),
                    "parent_path": str(rel_path),
                    "element_type": stype,
                    "selector": selector,
                    "image_path": rel_image_path,
                    "width": box['width'],
                    "height": box['height'],
                    "context_preview": context
                }
                metadata_list.append(meta)
                captured_count += 1
            except Exception as e:
                print(f"Failed to capture {rel_path} {selector} #{i}: {e}")

    return captured_count

async def process_file(sem: asyncio.Semaphore, browser, rel_path: str, metadata_list: List[Dict]):
    async with sem:
        page = await browser.new_page()
        try:
            url = f"{SERVER_URL}/{rel_path}"
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            await capture_elements(page, rel_path, metadata_list)
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
        finally:
            await page.close()

async def main():
    server_thread = threading.Thread(target=start_server, args=(PORT,), daemon=True)
    server_thread.start()
    await asyncio.sleep(2)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    html_files = get_html_files(DATA_DIR)
    print(f"Found {len(html_files)} HTML files. Starting async processing with concurrency {CONCURRENCY}...")
    
    metadata_list = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        sem = asyncio.Semaphore(CONCURRENCY)
        
        tasks = [process_file(sem, browser, f, metadata_list) for f in html_files]
        
        from tqdm.asyncio import tqdm
        for f in tqdm.as_completed(tasks):
            await f

        await browser.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        for entry in metadata_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Done. Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
