"""Smoke test for the corrector page (headless Chromium via Playwright).

    pip install playwright numpy soundfile python-docx
    TRANSCRIPT=/path/to/scribe.json python tools/corrector/smoke_test.py

TRANSCRIPT is an ElevenLabs Scribe JSON with word timestamps and speakers. It is never committed: the interviews are
consented research data. A short silent recording is generated here. The page loads JSZip from cdnjs; the test serves
a local copy if JSZIP=/path/to/jszip.min.js is set (npm pack jszip), otherwise the page's own CDN load is used.
"""
import io, json, os, sys, tempfile, zipfile
import numpy as np, soundfile as sf
from playwright.sync_api import sync_playwright

PAGE = "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "index.html"))
TRANSCRIPT = os.environ.get("TRANSCRIPT") or sys.exit("Set TRANSCRIPT=/path/to/scribe.json")
EXE = os.environ.get("CHROMIUM")          # e.g. /opt/pw-browsers/chromium-1194/chrome-linux/chrome
STUB = """window.claude={use:async(n)=>n==='downloads'?{save:async(r)=>{window.__saved={filename:r.filename};
window.__text=typeof r.data==='string'?r.data:null;window.__bytes=typeof r.data==='string'?null:Array.from(r.data);return {status:'saved'}}}:null};"""
failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + str(detail) if detail and not ok else ""))
    if not ok: failures.append(name)

tmp = tempfile.mkdtemp(); wav = os.path.join(tmp, "talk.wav")
sf.write(wav, np.zeros(8000 * 200, dtype="float32"), 8000, subtype="PCM_U8")
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]) if EXE else p.chromium.launch(args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 1180, "height": 800}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    if os.environ.get("JSZIP"): pg.route("**/jszip.min.js", lambda r: r.fulfill(body=open(os.environ["JSZIP"]).read(), content_type="application/javascript"))
    pg.add_init_script(STUB); pg.goto(PAGE)
    pg.set_input_files("#mediaFile", wav); pg.set_input_files("#jsonFile", TRANSCRIPT); pg.click("#startBtn"); pg.wait_for_selector("#work:not([hidden])")
    check("opens and builds the script", pg.locator("#prose .seg").count() > 100)
    check("editor focused", pg.evaluate("document.activeElement.id") == "text")
    pg.click("#bPlay"); check("Play keeps focus in the editor", pg.evaluate("document.activeElement.id") == "text")
    paused = lambda: pg.evaluate("document.getElementById('video').paused")
    pg.click('#prose .seg[data-i="3"]'); pg.wait_for_timeout(300); pg.keyboard.type("x")
    check("typing pauses the audio", paused())
    pg.fill("#text", "R: nibreakie and Mbagathi Hospital."); pg.evaluate("t=>{const a=document.getElementById('text');a.focus();a.setSelectionRange(3,5)}", 0); pg.click("#mItalic")
    check("italic on part of a word", pg.input_value("#text").startswith("R: _ni_break"), pg.input_value("#text"))
    pg.evaluate("()=>{const a=document.getElementById('text');const i=a.value.indexOf('Mbagathi');a.setSelectionRange(i,i+17)}"); pg.click("#mBold")
    check("bold toggles on a selection", "**Mbagathi Hospital**" in pg.input_value("#text"))
    pg.click("#bSave"); pg.wait_for_timeout(300)
    check("Save, play next chunk plays", not paused())
    pg.click("#bSaveGo"); pg.wait_for_timeout(1200)
    check("Save and continue keeps playing", not paused())
    pg.click("#tOut"); pg.fill("#xName", "TEST FILE"); pg.check("#xT1"); pg.click("#xDocx"); pg.wait_for_timeout(1500)
    data = bytes(pg.evaluate("window.__bytes") or [])
    z = zipfile.ZipFile(io.BytesIO(data)); doc = z.read("word/document.xml").decode()
    check("docx has header, footer fields, title", "TEST FILE" in z.read("word/header1.xml").decode() and "NUMPAGES" in z.read("word/footer1.xml").decode() and "TEST FILE" in doc)
    check("docx is Times New Roman 12 and keeps bold/italic", "Times New Roman" in doc and '<w:sz w:val="24"/>' in doc and "<w:i/>" in doc and "<w:b/>" in doc)
    pg.click("#xLog"); pg.wait_for_timeout(300); log = json.loads(pg.evaluate("window.__text"))
    check("corrections log lists the mixed word", any(m["written"] == "_ni_break" or m["written"].startswith("_ni_") for m in log["mixed_words"]), log["mixed_words"])
    # recordings list, language list and language labels
    pg.click("#tHome"); pg.wait_for_selector("#setup:not([hidden])")
    check("recording appears in the list with its audio kept", pg.locator(".rec").count() == 1 and "audio kept" in pg.inner_text(".rec"), pg.inner_text("#recList"))
    pg.click(".rec button:has-text('Continue')"); pg.wait_for_selector("#work:not([hidden])")
    check("Continue reopens it from the saved copy", pg.locator("#prose .seg").count() > 100)
    pg.click("#tLang"); pg.wait_for_timeout(300)
    check("languages panel lists words", pg.locator("#lBody .lrow").count() > 20, pg.locator("#lBody .lrow").count())
    unsure = pg.locator("#lBody .lrow:has(button:has-text('Swahili'))").first
    word = unsure.locator(".lw").inner_text(); unsure.locator("button:has-text('Swahili')").first.click(); pg.wait_for_timeout(300)
    lex = json.loads(pg.evaluate("localStorage.getItem('corrector-lexicon')"))
    check("a language answer is kept for every recording", lex["words"].get(word.lower()) == "sw", (word, lex))
    pg.click("#tOut"); pg.click("#xLang"); pg.wait_for_timeout(500); csv = pg.evaluate("window.__text")
    head = csv.split("\n")[0]
    check("language labels file has the columns", head == "recording,chunk,start,end,speaker,token,language,parts,source", head)
    check("labels include Swahili, English and your list", ",sw," in csv and ",en," in csv and "your list" in csv)
    # paste the transcript instead of choosing a file
    pg.click("#tHome"); pg.wait_for_selector("#setup:not([hidden])")
    pg.set_input_files("#mediaFile", wav); pg.fill("#pasteBox", '{"words": [{"text": "Habari", "start"'); pg.wait_for_timeout(700)
    check("incomplete paste is explained", "not complete JSON" in pg.inner_text("#pasteMsg") and pg.is_disabled("#startBtn"))
    pg.evaluate("""t => { const dt = new DataTransfer(); dt.setData('text', t); document.getElementById('pasteBox').dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true})); }""", open(TRANSCRIPT, encoding="utf-8").read()); pg.wait_for_timeout(1500)
    check("a big paste stays out of the box", len(pg.input_value("#pasteBox")) < 300, len(pg.input_value("#pasteBox")))
    check("a good paste is recognised", pg.inner_text("#pasteMsg").startswith("Looks right") and not pg.is_disabled("#startBtn"), pg.inner_text("#pasteMsg"))
    pg.click("#startBtn"); pg.wait_for_selector("#work:not([hidden])")
    check("opens from the pasted text", pg.locator("#prose .seg").count() > 100)
    # a role for every speaker, a swap for the whole recording
    pg.click("#tSpk"); pg.wait_for_timeout(200) if pg.is_hidden("#pSpk") else None
    check("two speakers, a role for each", pg.locator("#spkList .srow").count() == 2, pg.locator("#spkList .srow").count())
    pg.click('#prose .seg[data-i="30"]'); before = pg.input_value("#text")
    pg.click("#spkSwap"); pg.wait_for_timeout(300); after = pg.input_value("#text")
    flip = lambda t: "\n".join(("R" if l.startswith("I:") else "I" if l.startswith("R:") else l[:1]) + l[1:] for l in t.split("\n"))
    check("swap flips I: and R: everywhere", after == flip(before), (before[:60], after[:60]))
    pg.select_option("#spkList .srow:nth-child(1) select", "I"); pg.wait_for_timeout(300)
    check("two voices can share one role", "R:" not in pg.input_value("#text") and "I:" in pg.input_value("#text"), pg.input_value("#text")[:80])
    check("no page errors", not errs, errs)
    b.close()
print("\n%d failure(s)" % len(failures)); sys.exit(1 if failures else 0)
