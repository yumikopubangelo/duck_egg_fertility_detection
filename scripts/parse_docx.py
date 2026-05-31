import sys
import zipfile
import xml.etree.ElementTree as ET

def docx_to_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read('word/document.xml')
    except Exception as e:
        return f"ERROR reading {path}: {e}"
    try:
        root = ET.fromstring(xml)
    except Exception as e:
        return f"ERROR parsing XML in {path}: {e}"
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = []
    for p in root.findall('.//w:p', ns):
        texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n'.join(paragraphs)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parse_docx.py <file1.docx> [file2.docx ...]')
        sys.exit(1)
    for path in sys.argv[1:]:
        print('\n' + '='*80)
        print(f'FILE: {path}')
        print('='*80)
        print(docx_to_text(path))
