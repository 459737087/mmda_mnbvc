import xml.etree.ElementTree as ET

def xml_to_markdown(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Initialize a list to hold markdown elements
    markdown_elements = []

    # Add the initial text if it exists (text before the first subelement)
    if root.text:
        markdown_elements.append(root.text)
        print('root.text:'+root.text)
        print('markdown_elements:',markdown_elements)

    # Iterate over the XML elements inside the root
    for elem in root:
        # Check for 'ref' tag and append appropriate markdown
        if elem.tag == 'ref':
            ref_type = elem.attrib.get('type')
            if ref_type == 'bibr':
                # Convert bibliography reference to markdown citation format
                markdown_elements.append(f"[{elem.text}]")
                print('markdown_elements_bibr:',markdown_elements)
            elif ref_type == 'figure':
                # Convert figure reference to markdown image format (as a placeholder here)
                markdown_elements.append(f"![Figure {elem.text}]")
                print('markdown_elements_figure:',markdown_elements)
        # Append the tail text if it exists (text after a subelement)
        if elem.tail:
            print(elem.tail)
            markdown_elements.append(elem.tail)

    # Join the list into a single string and return it
    return ''.join(markdown_elements)

# XML string
xml_string = '''
<s coords="1,173.74,668.65,112.63,8.64;1,50.11,680.60,236.25,8.64;1,50.11,692.56,236.25,8.64;1,50.11,704.51,236.25,8.64;1,308.86,401.32,138.54,8.64">
Since the data provides convenient and large-scale coverage, people are using it for a number of societally important problems such as traffic monitoring <ref type="bibr" coords="1,97.49,704.51,15.27,8.64" target="#b20">[21]</ref>, urban planning <ref type="bibr" coords="1,183.57,704.51,10.58,8.64" target="#b3">[4]</ref>, vehicle detection <ref type="bibr" coords="1,272.26,704.51,10.58,8.64" target="#b8">[9]</ref>, and Figure <ref type="figure" coords="1,337.94,401.32,3.88,8.64">1</ref>: Motivation of our work.
</s>
'''

# Convert XML to Markdown
markdown_output = xml_to_markdown(xml_string)
print(markdown_output)