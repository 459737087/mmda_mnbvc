import xml.etree.ElementTree as ET

def xml_to_markdown(root,namespaces):
    markdown_list = []

    # Iterate over 'biblStruct' elements inside 'listBibl'
    for bibl_index, bibl_struct in enumerate(root.findall(".//tei:biblStruct", namespaces=namespaces), start=1):
        # Extract the title and URL from either monogr or analytic part
        title_text = ''
        url = ''
        monogr = bibl_struct.find(".//tei:monogr", namespaces=namespaces)
        analytic = bibl_struct.find(".//tei:analytic", namespaces=namespaces)

        if analytic is not None:
            title = analytic.find(".//tei:title[@level='a']", namespaces=namespaces)
            if title is not None:
                title_text = title.text.strip()
        elif monogr is not None:
            title = monogr.find(".//tei:title[@level='m']", namespaces=namespaces)
            if title is not None:
                title_text = title.text.strip()
            ptr = monogr.find(".//tei:ptr", namespaces=namespaces)
            if ptr is not None:
                url = ptr.attrib.get('target', '')

        # Extract author information
        authors = bibl_struct.findall(".//tei:author/persName", namespaces=namespaces)
        author_names = []
        for author in authors:
            forename = author.find('tei:forename', namespaces=namespaces).text if author.find('tei:forename', namespaces=namespaces) is not None else ''
            surname = author.find('tei:surname', namespaces=namespaces).text if author.find('tei:surname', namespaces=namespaces) is not None else ''
            author_names.append(f"{forename.strip()}. {surname.strip()}")

        # Extract additional publication information
        journal_name = bibl_struct.findtext(".//tei:title[@level='j']", namespaces=namespaces)
        issue = bibl_struct.findtext(".//tei:biblScope[@unit='issue']", namespaces=namespaces)
        date = bibl_struct.findtext(".//tei:date[@type='published']", namespaces=namespaces)

        # Construct the markdown reference
        author_str = ', '.join(author_names)
        markdown_ref = f"[{bibl_index}] {author_str}. {title_text}."
        if journal_name:
            markdown_ref += f" {journal_name},"
        if date:
            markdown_ref += f" {date}."
        if issue:
            markdown_ref += f" {issue}"

        if url:
            markdown_ref += f" [{url}]({url})"

        markdown_list.append(markdown_ref)

    # Join all references with new lines
    markdown_output = "\n".join(markdown_list)
    markdown_output = "## References" + "\n\n" + markdown_output
    return markdown_output
