from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
tree = etree.parse('/tmp/doc1/word/document.xml')
body = tree.find('{%s}body' % W)
children = list(body)
print('Total body children:', len(children))
tbl_tag = '{%s}tbl' % W
for i in range(len(children)-1):
    t1 = children[i].tag.split('}')[-1]
    t2 = children[i+1].tag.split('}')[-1]
    if t1 == 'tbl' and t2 == 'tbl':
        print('Adjacent tables at', i, i+1)
for i, el in enumerate(children):
    if el.tag == tbl_tag:
        parent_tag = el.getparent().tag.split('}')[-1]
        if parent_tag == 'body':
            pass
        else:
            print('Tbl at', i, 'has parent:', parent_tag)
print("done")
