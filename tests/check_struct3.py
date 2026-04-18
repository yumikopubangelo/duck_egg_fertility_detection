from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
tree = etree.parse('/tmp/doc1/word/document.xml')
body = tree.find('{%s}body' % W)
children = list(body)
print('Total body children:', len(children))
# Print all tag names with index
for i, el in enumerate(children):
    tag = el.tag.split('}')[-1]
    txt = ''.join(t.text or '' for t in el.iter('{%s}t' % W))[:60]
    print(i, tag, txt[:50])
