"""Fix the combined document: move mid-body sectPr into paragraph pPr."""
from lxml import etree
import copy

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
tree = etree.parse('/tmp/doc1/word/document.xml')
body = tree.find('{%s}body' % W)
children = list(body)

sect_tag = '{%s}sectPr' % W
p_tag    = '{%s}p'      % W
pPr_tag  = '{%s}pPr'    % W

# Find all body-level sectPr elements that are NOT the last child
last_child = children[-1]
to_fix = []
for el in children:
    if el.tag == sect_tag and el is not last_child:
        to_fix.append(el)

print('Found', len(to_fix), 'mid-body sectPr to wrap')

for sect_el in to_fix:
    # Create a wrapper paragraph
    p_wrap = etree.Element(p_tag)
    pPr = etree.SubElement(p_wrap, pPr_tag)
    pPr.append(copy.deepcopy(sect_el))
    # Insert the wrapper paragraph before the sectPr
    parent = sect_el.getparent()
    idx = list(parent).index(sect_el)
    parent.insert(idx, p_wrap)
    parent.remove(sect_el)

# Write back
with open('/tmp/doc1/word/document.xml', 'wb') as f:
    f.write(etree.tostring(tree, xml_declaration=True,
                           encoding='UTF-8', standalone=True))
print('Fixed and written.')
