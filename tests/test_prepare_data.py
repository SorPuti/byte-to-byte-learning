from prepare_data import strip_gutenberg

HEADER = "Project Gutenberg header line\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
FOOTER = "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense text\n"


def test_strip_keeps_only_body():
    text = HEADER + "Capítulo primeiro.\nSegunda linha.\n" + FOOTER
    assert strip_gutenberg(text) == "Capítulo primeiro.\nSegunda linha.\n"


def test_strip_without_markers_returns_text():
    assert strip_gutenberg("apenas texto\n") == "apenas texto\n"
